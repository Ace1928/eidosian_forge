from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import stat
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError
from ansible.executor.playbook_executor import PlaybookExecutor
from ansible.module_utils.common.text.converters import to_bytes
from ansible.playbook.block import Block
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.utils.display import Display
class PlaybookCLI(CLI):
    """ the tool to run *Ansible playbooks*, which are a configuration and multinode deployment system.
        See the project home page (https://docs.ansible.com) for more information. """
    name = 'ansible-playbook'

    def init_parser(self):
        super(PlaybookCLI, self).init_parser(usage='%prog [options] playbook.yml [playbook2 ...]', desc='Runs Ansible playbooks, executing the defined tasks on the targeted hosts.')
        opt_help.add_connect_options(self.parser)
        opt_help.add_meta_options(self.parser)
        opt_help.add_runas_options(self.parser)
        opt_help.add_subset_options(self.parser)
        opt_help.add_check_options(self.parser)
        opt_help.add_inventory_options(self.parser)
        opt_help.add_runtask_options(self.parser)
        opt_help.add_vault_options(self.parser)
        opt_help.add_fork_options(self.parser)
        opt_help.add_module_options(self.parser)
        self.parser.add_argument('--syntax-check', dest='syntax', action='store_true', help='perform a syntax check on the playbook, but do not execute it')
        self.parser.add_argument('--list-tasks', dest='listtasks', action='store_true', help='list all tasks that would be executed')
        self.parser.add_argument('--list-tags', dest='listtags', action='store_true', help='list all available tags')
        self.parser.add_argument('--step', dest='step', action='store_true', help='one-step-at-a-time: confirm each task before running')
        self.parser.add_argument('--start-at-task', dest='start_at_task', help='start the playbook at the task matching this name')
        self.parser.add_argument('args', help='Playbook(s)', metavar='playbook', nargs='+')

    def post_process_args(self, options):
        havetags = bool(options.tags or options.skip_tags)
        options = super(PlaybookCLI, self).post_process_args(options)
        if options.listtags:
            if not havetags:
                options.tags = ['never', 'all']
        display.verbosity = options.verbosity
        self.validate_conflicts(options, runas_opts=True, fork_opts=True)
        return options

    def run(self):
        super(PlaybookCLI, self).run()
        sshpass = None
        becomepass = None
        passwords = {}
        b_playbook_dirs = []
        for playbook in context.CLIARGS['args']:
            resource = _get_collection_playbook_path(playbook)
            if resource is not None:
                playbook_collection = resource[2]
            else:
                if not os.path.exists(playbook):
                    raise AnsibleError('the playbook: %s could not be found' % playbook)
                if not (os.path.isfile(playbook) or stat.S_ISFIFO(os.stat(playbook).st_mode)):
                    raise AnsibleError('the playbook: %s does not appear to be a file' % playbook)
                playbook_collection = _get_collection_name_from_path(playbook)
            if not playbook_collection:
                b_playbook_dir = os.path.dirname(os.path.abspath(to_bytes(playbook, errors='surrogate_or_strict')))
                add_all_plugin_dirs(b_playbook_dir)
                b_playbook_dirs.append(b_playbook_dir)
        if b_playbook_dirs:
            AnsibleCollectionConfig.playbook_paths = b_playbook_dirs
        if not (context.CLIARGS['listhosts'] or context.CLIARGS['listtasks'] or context.CLIARGS['listtags'] or context.CLIARGS['syntax']):
            sshpass, becomepass = self.ask_passwords()
            passwords = {'conn_pass': sshpass, 'become_pass': becomepass}
        loader, inventory, variable_manager = self._play_prereqs()
        CLI.get_host_list(inventory, context.CLIARGS['subset'])
        if context.CLIARGS['flush_cache']:
            self._flush_cache(inventory, variable_manager)
        pbex = PlaybookExecutor(playbooks=context.CLIARGS['args'], inventory=inventory, variable_manager=variable_manager, loader=loader, passwords=passwords)
        results = pbex.run()
        if isinstance(results, list):
            for p in results:
                display.display('\nplaybook: %s' % p['playbook'])
                for idx, play in enumerate(p['plays']):
                    if play._included_path is not None:
                        loader.set_basedir(play._included_path)
                    else:
                        pb_dir = os.path.realpath(os.path.dirname(p['playbook']))
                        loader.set_basedir(pb_dir)
                    try:
                        host_list = ','.join(play.hosts)
                    except TypeError:
                        host_list = ''
                    msg = '\n  play #%d (%s): %s' % (idx + 1, host_list, play.name)
                    mytags = set(play.tags)
                    msg += '\tTAGS: [%s]' % ','.join(mytags)
                    if context.CLIARGS['listhosts']:
                        playhosts = set(inventory.get_hosts(play.hosts))
                        msg += '\n    pattern: %s\n    hosts (%d):' % (play.hosts, len(playhosts))
                        for host in playhosts:
                            msg += '\n      %s' % host
                    display.display(msg)
                    all_tags = set()
                    if context.CLIARGS['listtags'] or context.CLIARGS['listtasks']:
                        taskmsg = ''
                        if context.CLIARGS['listtasks']:
                            taskmsg = '    tasks:\n'

                        def _process_block(b):
                            taskmsg = ''
                            for task in b.block:
                                if isinstance(task, Block):
                                    taskmsg += _process_block(task)
                                else:
                                    if task.action in C._ACTION_META and task.implicit:
                                        continue
                                    all_tags.update(task.tags)
                                    if context.CLIARGS['listtasks']:
                                        cur_tags = list(mytags.union(set(task.tags)))
                                        cur_tags.sort()
                                        if task.name:
                                            taskmsg += '      %s' % task.get_name()
                                        else:
                                            taskmsg += '      %s' % task.action
                                        taskmsg += '\tTAGS: [%s]\n' % ', '.join(cur_tags)
                            return taskmsg
                        all_vars = variable_manager.get_vars(play=play)
                        for block in play.compile():
                            block = block.filter_tagged_tasks(all_vars)
                            if not block.has_tasks():
                                continue
                            taskmsg += _process_block(block)
                        if context.CLIARGS['listtags']:
                            cur_tags = list(mytags.union(all_tags))
                            cur_tags.sort()
                            taskmsg += '      TASK TAGS: [%s]\n' % ', '.join(cur_tags)
                        display.display(taskmsg)
            return 0
        else:
            return results

    @staticmethod
    def _flush_cache(inventory, variable_manager):
        for host in inventory.list_hosts():
            hostname = host.get_name()
            variable_manager.clear_facts(hostname)