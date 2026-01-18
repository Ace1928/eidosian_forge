from libcloud.common.types import (
class DeploymentError(LibcloudError):
    """
    Exception used when a Deployment Task failed.

    :ivar node: :class:`Node` on which this exception happened, you might want
                to call :func:`Node.destroy`
    """

    def __init__(self, node, original_exception=None, driver=None):
        self.node = node
        self.value = original_exception
        self.original_error = original_exception
        self.driver = driver

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<DeploymentError: node={}, error={}, driver={}>'.format(self.node.id, str(self.value), str(self.driver))