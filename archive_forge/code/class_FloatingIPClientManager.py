from blazarclient import base
class FloatingIPClientManager(base.BaseClientManager):
    """Manager for floating IP requests."""

    def create(self, network_id, floating_ip_address, **kwargs):
        """Creates a floating IP from values passed."""
        values = {'floating_network_id': network_id, 'floating_ip_address': floating_ip_address}
        values.update(**kwargs)
        resp, body = self.request_manager.post('/floatingips', body=values)
        return body['floatingip']

    def get(self, floatingip_id):
        """Show floating IP details."""
        resp, body = self.request_manager.get('/floatingips/%s' % floatingip_id)
        return body['floatingip']

    def delete(self, floatingip_id):
        """Deletes floating IP with specified ID."""
        resp, body = self.request_manager.delete('/floatingips/%s' % floatingip_id)

    def list(self, sort_by=None):
        """List all floating IPs."""
        resp, body = self.request_manager.get('/floatingips')
        floatingips = body['floatingips']
        if sort_by:
            floatingips = sorted(floatingips, key=lambda l: l[sort_by])
        return floatingips