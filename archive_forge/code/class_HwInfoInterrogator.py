from troveclient import base
class HwInfoInterrogator(base.ManagerWithFind):
    """Manager class for HwInfo."""
    resource_class = HwInfo

    def get(self, instance):
        """Get the hardware information of the instance."""
        return self._get('/mgmt/instances/%s/hwinfo' % base.getid(instance))

    def list(self):
        pass