from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitLabHook(object):

    def __init__(self, module, gitlab_instance):
        self._module = module
        self._gitlab = gitlab_instance
        self.hook_object = None
    '\n    @param project Project Object\n    @param hook_url Url to call on event\n    @param description Description of the group\n    @param parent Parent group full path\n    '

    def create_or_update_hook(self, project, hook_url, options):
        changed = False
        if self.hook_object is None:
            hook = self.create_hook(project, {'url': hook_url, 'push_events': options['push_events'], 'push_events_branch_filter': options['push_events_branch_filter'], 'issues_events': options['issues_events'], 'merge_requests_events': options['merge_requests_events'], 'tag_push_events': options['tag_push_events'], 'note_events': options['note_events'], 'job_events': options['job_events'], 'pipeline_events': options['pipeline_events'], 'wiki_page_events': options['wiki_page_events'], 'releases_events': options['releases_events'], 'enable_ssl_verification': options['enable_ssl_verification'], 'token': options['token']})
            changed = True
        else:
            changed, hook = self.update_hook(self.hook_object, {'push_events': options['push_events'], 'push_events_branch_filter': options['push_events_branch_filter'], 'issues_events': options['issues_events'], 'merge_requests_events': options['merge_requests_events'], 'tag_push_events': options['tag_push_events'], 'note_events': options['note_events'], 'job_events': options['job_events'], 'pipeline_events': options['pipeline_events'], 'wiki_page_events': options['wiki_page_events'], 'releases_events': options['releases_events'], 'enable_ssl_verification': options['enable_ssl_verification'], 'token': options['token']})
        self.hook_object = hook
        if changed:
            if self._module.check_mode:
                self._module.exit_json(changed=True, msg='Successfully created or updated the hook %s' % hook_url)
            try:
                hook.save()
            except Exception as e:
                self._module.fail_json(msg='Failed to update hook: %s ' % e)
        return changed
    '\n    @param project Project Object\n    @param arguments Attributes of the hook\n    '

    def create_hook(self, project, arguments):
        if self._module.check_mode:
            return True
        hook = project.hooks.create(arguments)
        return hook
    '\n    @param hook Hook Object\n    @param arguments Attributes of the hook\n    '

    def update_hook(self, hook, arguments):
        changed = False
        for arg_key, arg_value in arguments.items():
            if arg_value is not None:
                if getattr(hook, arg_key, None) != arg_value:
                    setattr(hook, arg_key, arg_value)
                    changed = True
        return (changed, hook)
    '\n    @param project Project object\n    @param hook_url Url to call on event\n    '

    def find_hook(self, project, hook_url):
        for hook in project.hooks.list(**list_all_kwargs):
            if hook.url == hook_url:
                return hook
    '\n    @param project Project object\n    @param hook_url Url to call on event\n    '

    def exists_hook(self, project, hook_url):
        hook = self.find_hook(project, hook_url)
        if hook:
            self.hook_object = hook
            return True
        return False

    def delete_hook(self):
        if not self._module.check_mode:
            self.hook_object.delete()