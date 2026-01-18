from cinderclient import api_versions
from cinderclient import base
class VolumeAttachmentManager(base.ManagerWithFind):
    resource_class = VolumeAttachment

    @api_versions.wraps('3.27')
    def create(self, volume_id, connector, instance_id=None, mode='null'):
        """Create a attachment for specified volume."""
        body = {'attachment': {'volume_uuid': volume_id, 'connector': connector}}
        if instance_id:
            body['attachment']['instance_uuid'] = instance_id
        if self.api_version >= api_versions.APIVersion('3.54'):
            if mode and mode != 'null':
                body['attachment']['mode'] = mode
        retval = self._create('/attachments', body, 'attachment')
        return retval.to_dict()

    @api_versions.wraps('3.27')
    def delete(self, attachment):
        """Delete an attachment by ID."""
        return self._delete('/attachments/%s' % base.getid(attachment))

    @api_versions.wraps('3.27')
    def list(self, detailed=False, search_opts=None, marker=None, limit=None, sort=None):
        """List all attachments."""
        resource_type = 'attachments'
        url = self._build_list_url(resource_type, detailed=detailed, search_opts=search_opts, marker=marker, limit=limit, sort=sort)
        return self._list(url, resource_type, limit=limit)

    @api_versions.wraps('3.27')
    def show(self, id):
        """Attachment show.

        :param id: Attachment ID.
        """
        url = '/attachments/%s' % id
        resp, body = self.api.client.get(url)
        return self.resource_class(self, body['attachment'], loaded=True, resp=resp)

    @api_versions.wraps('3.27')
    def update(self, id, connector):
        """Attachment update."""
        body = {'attachment': {'connector': connector}}
        resp = self._update('/attachments/%s' % id, body)
        return self.resource_class(self, resp['attachment'], loaded=True, resp=resp)

    @api_versions.wraps('3.44')
    def complete(self, attachment):
        """Mark the attachment as completed."""
        resp, body = self._action_return_resp_and_body('os-complete', attachment, None)
        return resp

    def _action_return_resp_and_body(self, action, attachment, info=None, **kwargs):
        """Perform a attachments "action" and return response headers and body.

        """
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/attachments/%s/action' % base.getid(attachment)
        return self.api.client.post(url, body=body)