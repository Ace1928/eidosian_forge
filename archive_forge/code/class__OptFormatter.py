import collections
import copy
import importlib.metadata
import json
import logging
import operator
import sys
import yaml
from oslo_config import cfg
from oslo_i18n import _message
import stevedore.named  # noqa
class _OptFormatter:
    """Format configuration option descriptions to a file."""

    def __init__(self, conf, output_file=None):
        """Construct an OptFormatter object.

        :param conf: The config object from _generator_opts
        :param output_file: a writeable file object
        """
        self.output_file = output_file or sys.stdout
        self.wrap_width = conf.wrap_width or 998
        self.minimal = conf.minimal
        self.summarize = conf.summarize
        if rst2txt:
            for rolename, nodeclass in sphinx_roles.generic_docroles.items():
                generic = docutils_roles.GenericRole(rolename, nodeclass)
                docutils_roles.register_local_role(rolename, generic)
            for rolename, func in sphinx_roles.specific_docroles.items():
                docutils_roles.register_local_role(rolename, func)
            for rolename in ('oslo.config:option', 'oslo.config:group'):
                generic = docutils_roles.GenericRole(rolename, docutils_nodes.strong)
                docutils_roles.register_local_role(rolename, generic)

    def _format_help(self, help_text):
        """Format the help for a group or option to the output file.

        :param help_text: The text of the help string
        """
        if rst2txt:
            help_text = docutils_core.publish_string(source=help_text, writer=rst2txt.Writer(), settings_overrides={'wrap_width': self.wrap_width}).decode()
            lines = ''
            for line in help_text.splitlines():
                lines += '# ' + line + '\n' if line else '#\n'
            lines = [lines]
        elif self.wrap_width > 0:
            wrapped = ''
            for line in help_text.splitlines():
                text = '\n'.join(textwrap.wrap(line, self.wrap_width, initial_indent='# ', subsequent_indent='# ', break_long_words=False, replace_whitespace=False))
                wrapped += '#' if text == '' else text
                wrapped += '\n'
            lines = [wrapped]
        else:
            lines = ['# ' + help_text + '\n']
        return lines

    def _get_choice_text(self, choice):
        if choice is None:
            return '<None>'
        elif choice == '':
            return "''"
        return str(choice)

    def format_group(self, group_or_groupname):
        """Format the description of a group header to the output file

        :param group_or_groupname: a cfg.OptGroup instance or a name of group
        :returns: a formatted group description string
        """
        if isinstance(group_or_groupname, cfg.OptGroup):
            group = group_or_groupname
            lines = ['[%s]\n' % group.name]
            if group.help:
                lines += self._format_help(group.help)
        else:
            groupname = group_or_groupname
            lines = ['[%s]\n' % groupname]
        self.writelines(lines)

    def format(self, opt, group_name):
        """Format a description of an option to the output file.

        :param opt: a cfg.Opt instance
        :param group_name: name of the group to which the opt is assigned
        :returns: a formatted opt description string
        """
        if not opt.help:
            LOG.warning('"%s" is missing a help string', opt.dest)
        opt_type = _format_type_name(opt.type)
        opt_prefix = ''
        if opt.deprecated_for_removal and (not opt.help.startswith('DEPRECATED')):
            opt_prefix = 'DEPRECATED: '
        if opt.help:
            if self.summarize:
                _split = opt.help.split('\n\n')
                opt_help = _split[0].rstrip(':').rstrip('.')
                if len(_split) > 1:
                    opt_help += '. For more information, refer to the '
                    opt_help += 'documentation.'
            else:
                opt_help = opt.help
            help_text = '%s%s (%s)' % (opt_prefix, opt_help, opt_type)
        else:
            help_text = '(%s)' % opt_type
        lines = self._format_help(help_text)
        if getattr(opt.type, 'min', None) is not None:
            lines.append('# Minimum value: {}\n'.format(opt.type.min))
        if getattr(opt.type, 'max', None) is not None:
            lines.append('# Maximum value: {}\n'.format(opt.type.max))
        if getattr(opt.type, 'choices', None):
            lines.append('# Possible values:\n')
            for choice in opt.type.choices:
                help_text = '%s - %s' % (self._get_choice_text(choice), opt.type.choices[choice] or '<No description provided>')
                lines.extend(self._format_help(help_text))
        try:
            if opt.mutable:
                lines.append('# Note: This option can be changed without restarting.\n')
        except AttributeError as err:
            import warnings
            if not isinstance(opt, cfg.Opt):
                warnings.warn('Incompatible option class for %s (%r): %s' % (opt.dest, opt.__class__, err))
            else:
                warnings.warn('Failed to fully format sample for %s: %s' % (opt.dest, err))
        for d in opt.deprecated_opts:
            if d.name and '-' not in d.name:
                lines.append('# Deprecated group/name - [%s]/%s\n' % (d.group or group_name, d.name or opt.dest))
        if opt.deprecated_for_removal:
            if opt.deprecated_since:
                lines.append('# This option is deprecated for removal since %s.\n' % opt.deprecated_since)
            else:
                lines.append('# This option is deprecated for removal.\n')
            lines.append('# Its value may be silently ignored in the future.\n')
            if opt.deprecated_reason:
                lines.extend(self._format_help('Reason: ' + opt.deprecated_reason))
        if opt.advanced:
            lines.append('# Advanced Option: intended for advanced users and not used\n# by the majority of users, and might have a significant\n# effect on stability and/or performance.\n')
        if opt.sample_default:
            lines.append('#\n# This option has a sample default set, which means that\n# its actual default value may vary from the one documented\n# below.\n')
        if hasattr(opt.type, 'format_defaults'):
            defaults = opt.type.format_defaults(opt.default, opt.sample_default)
        else:
            LOG.debug("The type for option %(name)s which is %(type)s is not a subclass of types.ConfigType and doesn't provide a 'format_defaults' method. A default formatter is not available so the best-effort formatter will be used.", {'type': opt.type, 'name': opt.name})
            defaults = _format_defaults(opt)
        for default_str in defaults:
            if default_str:
                default_str = ' ' + default_str.replace('\n', '\n#    ')
            if self.minimal:
                lines.append('%s =%s\n' % (opt.dest, default_str))
            else:
                lines.append('#%s =%s\n' % (opt.dest, default_str))
        self.writelines(lines)

    def write(self, s):
        """Write an arbitrary string to the output file.

        :param s: an arbitrary string
        """
        self.output_file.write(s)

    def writelines(self, lines):
        """Write an arbitrary sequence of strings to the output file.

        :param lines: a list of arbitrary strings
        """
        self.output_file.writelines(lines)