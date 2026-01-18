import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
class DataCollector:

    def __init__(self, no_plugins=False, selected_plugins=None):
        self.data = CompletionData()
        self.user_aliases = {}
        if no_plugins:
            self.selected_plugins = set()
        elif selected_plugins is None:
            self.selected_plugins = None
        else:
            self.selected_plugins = {x.replace('-', '_') for x in selected_plugins}

    def collect(self):
        self.global_options()
        self.aliases()
        self.commands()
        return self.data

    def global_options(self):
        for name, item in option.Option.OPTIONS.items():
            self.data.global_options.append(('--' + item.name, '-' + item.short_name() if item.short_name() else None, item.help.rstrip()))

    def aliases(self):
        for alias, expansion in config.GlobalConfig().get_aliases().items():
            for token in cmdline.split(expansion):
                if not token.startswith('-'):
                    self.user_aliases.setdefault(token, set()).add(alias)
                    break

    def commands(self):
        for name in sorted(commands.all_command_names()):
            self.command(name)

    def command(self, name):
        cmd = commands.get_cmd_object(name)
        cmd_data = CommandData(name)
        plugin_name = cmd.plugin_name()
        if plugin_name is not None:
            if self.selected_plugins is not None and plugin not in self.selected_plugins:
                return None
            plugin_data = self.data.plugins.get(plugin_name)
            if plugin_data is None:
                plugin_data = PluginData(plugin_name)
                self.data.plugins[plugin_name] = plugin_data
            cmd_data.plugin = plugin_data
        self.data.commands.append(cmd_data)
        cmd_data.aliases.extend(cmd.aliases)
        cmd_data.aliases.extend(sorted([useralias for cmdalias in cmd_data.aliases if cmdalias in self.user_aliases for useralias in self.user_aliases[cmdalias] if useralias not in cmd_data.aliases]))
        opts = cmd.options()
        for optname, opt in sorted(opts.items()):
            cmd_data.options.extend(self.option(opt))
        if 'help' == name or 'help' in cmd.aliases:
            cmd_data.fixed_words = '($cmds %s)' % ' '.join(sorted(help_topics.topic_registry.keys()))
        return cmd_data

    def option(self, opt):
        optswitches = {}
        parser = option.get_optparser([opt])
        parser = self.wrap_parser(optswitches, parser)
        optswitches.clear()
        opt.add_option(parser, opt.short_name())
        if isinstance(opt, option.RegistryOption) and opt.enum_switch:
            enum_switch = '--%s' % opt.name
            enum_data = optswitches.get(enum_switch)
            if enum_data:
                try:
                    enum_data.registry_keys = opt.registry.keys()
                except ImportError as e:
                    enum_data.error_messages.append("ERROR getting registry keys for '--%s': %s" % (opt.name, str(e).split('\n')[0]))
        return sorted(optswitches.values())

    def wrap_container(self, optswitches, parser):

        def tweaked_add_option(*opts, **attrs):
            for name in opts:
                optswitches[name] = OptionData(name)
        parser.add_option = tweaked_add_option
        return parser

    def wrap_parser(self, optswitches, parser):
        orig_add_option_group = parser.add_option_group

        def tweaked_add_option_group(*opts, **attrs):
            return self.wrap_container(optswitches, orig_add_option_group(*opts, **attrs))
        parser.add_option_group = tweaked_add_option_group
        return self.wrap_container(optswitches, parser)