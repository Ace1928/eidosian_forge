from urllib import parse
from heatclient._i18n import _
from heatclient.common import base
from heatclient.common import utils
from heatclient import exc
class StackManager(StackChildManager):
    resource_class = Stack

    def list(self, **kwargs):
        """Get a list of stacks.

        :param limit: maximum number of stacks to return
        :param marker: begin returning stacks that appear later in the stack
                       list than that represented by this stack id
        :param filters: dict of direct comparison filters that mimics the
                        structure of a stack object
        :rtype: list of :class:`Stack`
        """

        def paginate(params):
            """Paginate stacks, even if more than API limit."""
            current_limit = int(params.get('limit') or 0)
            url = '/stacks?%s' % parse.urlencode(params, True)
            stacks = self._list(url, 'stacks')
            for stack in stacks:
                yield stack
            num_stacks = len(stacks)
            remaining_limit = current_limit - num_stacks
            if remaining_limit > 0 and num_stacks > 0:
                params['limit'] = remaining_limit
                params['marker'] = stack.id
                for stack in paginate(params):
                    yield stack
        params = {}
        if 'filters' in kwargs:
            filters = kwargs.pop('filters')
            params.update(filters)
        for key, value in kwargs.items():
            if value:
                params[key] = value
        return paginate(params)

    def preview(self, **kwargs):
        """Preview a stack."""
        headers = self.client.credentials_headers()
        resp = self.client.post('/stacks/preview', data=kwargs, headers=headers)
        body = utils.get_response_body(resp)
        return Stack(self, body.get('stack'))

    def create(self, **kwargs):
        """Create a stack."""
        headers = self.client.credentials_headers()
        resp = self.client.post('/stacks', data=kwargs, headers=headers)
        body = utils.get_response_body(resp)
        return body

    def update(self, stack_id, **kwargs):
        """Update a stack.

        :param stack_id: Stack name or ID to identifies the stack
        """
        headers = self.client.credentials_headers()
        if kwargs.pop('existing', None):
            self.client.patch('/stacks/%s' % stack_id, data=kwargs, headers=headers)
        else:
            self.client.put('/stacks/%s' % stack_id, data=kwargs, headers=headers)

    def preview_update(self, stack_id, **kwargs):
        """Preview a stack update.

        :param stack_id: Stack name or ID to identifies the stack
        """
        stack_identifier = self._resolve_stack_id(stack_id)
        headers = self.client.credentials_headers()
        path = '/stacks/%s/preview' % stack_identifier
        if kwargs.pop('show_nested', False):
            path += '?show_nested=True'
        if kwargs.pop('existing', None):
            resp = self.client.patch(path, data=kwargs, headers=headers)
        else:
            resp = self.client.put(path, data=kwargs, headers=headers)
        body = utils.get_response_body(resp)
        return body

    def delete(self, stack_id):
        """Delete a stack.

        :param stack_id: Stack name or ID to identifies the stack
        """
        self._delete('/stacks/%s' % stack_id)

    def abandon(self, stack_id):
        """Abandon a stack.

        :param stack_id: Stack name or ID to identifies the stack
        """
        stack_identifier = self._resolve_stack_id(stack_id)
        resp = self.client.delete('/stacks/%s/abandon' % stack_identifier)
        body = utils.get_response_body(resp)
        return body

    def export(self, stack_id):
        """Export data of a stack.

        :param stack_id: Stack name or ID to identifies the stack
        """
        stack_identifier = self._resolve_stack_id(stack_id)
        resp = self.client.get('/stacks/%s/export' % stack_identifier)
        body = utils.get_response_body(resp)
        return body

    def snapshot(self, stack_id, name=None):
        """Snapshot a stack.

        :param stack_id: Stack name or ID to identifies the stack
        """
        stack_identifier = self._resolve_stack_id(stack_id)
        data = {}
        if name:
            data['name'] = name
        resp = self.client.post('/stacks/%s/snapshots' % stack_identifier, data=data)
        body = utils.get_response_body(resp)
        return body

    def snapshot_show(self, stack_id, snapshot_id):
        stack_identifier = self._resolve_stack_id(stack_id)
        resp = self.client.get('/stacks/%s/snapshots/%s' % (stack_identifier, snapshot_id))
        body = utils.get_response_body(resp)
        return body

    def snapshot_delete(self, stack_id, snapshot_id):
        stack_identifier = self._resolve_stack_id(stack_id)
        resp = self.client.delete('/stacks/%s/snapshots/%s' % (stack_identifier, snapshot_id))
        body = utils.get_response_body(resp)
        return body

    def restore(self, stack_id, snapshot_id):
        stack_identifier = self._resolve_stack_id(stack_id)
        resp = self.client.post('/stacks/%s/snapshots/%s/restore' % (stack_identifier, snapshot_id))
        body = utils.get_response_body(resp)
        return body

    def snapshot_list(self, stack_id):
        stack_identifier = self._resolve_stack_id(stack_id)
        resp = self.client.get('/stacks/%s/snapshots' % stack_identifier)
        body = utils.get_response_body(resp)
        return body

    def output_list(self, stack_id):
        stack_identifier = self._resolve_stack_id(stack_id)
        resp = self.client.get('/stacks/%s/outputs' % stack_identifier)
        body = utils.get_response_body(resp)
        return body

    def output_show(self, stack_id, output_key):
        stack_identifier = self._resolve_stack_id(stack_id)
        resp = self.client.get('/stacks/%(id)s/outputs/%(key)s' % {'id': stack_identifier, 'key': output_key})
        body = utils.get_response_body(resp)
        return body

    def get(self, stack_id, resolve_outputs=True):
        """Get the metadata for a specific stack.

        :param stack_id: Stack ID or name to lookup
        :param resolve_outputs: If True, then outputs for this
               stack will be resolved
        """
        kwargs = {}
        if not resolve_outputs:
            kwargs['params'] = {'resolve_outputs': False}
        resp = self.client.get('/stacks/%s' % stack_id, **kwargs)
        body = utils.get_response_body(resp)
        return Stack(self, body.get('stack'), loaded=True)

    def template(self, stack_id):
        """Get template content for a specific stack as a parsed JSON object.

        :param stack_id: Stack ID or name to get the template for
        """
        resp = self.client.get('/stacks/%s/template' % stack_id)
        body = utils.get_response_body(resp)
        return body

    def environment(self, stack_id):
        """Returns the environment for an existing stack.

        :param stack_id: Stack name or ID to identifies the stack
        :return:
        """
        resp = self.client.get('/stacks/%s/environment' % stack_id)
        body = utils.get_response_body(resp)
        return body

    def files(self, stack_id):
        """Returns the files for an existing stack.

        :param stack_id: Stack name or ID to identifies the stack.
        :return:
        """
        resp = self.client.get('/stacks/%s/files' % stack_id)
        body = utils.get_response_body(resp)
        return body

    def validate(self, **kwargs):
        """Validate a stack template."""
        url = '/validate'
        params = {}
        if kwargs.pop('show_nested', False):
            params['show_nested'] = True
        ignore_errors = kwargs.pop('ignore_errors', None)
        if ignore_errors:
            params['ignore_errors'] = ignore_errors
        args = {}
        if kwargs:
            args['data'] = kwargs
        if params:
            args['params'] = params
        resp = self.client.post(url, **args)
        body = utils.get_response_body(resp)
        return body