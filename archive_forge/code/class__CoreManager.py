from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import ActivityException
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborsConf
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfsConf
class _CoreManager(Activity):
    """Core service manager.
    """

    def __init__(self):
        self._common_conf = None
        self._neighbors_conf = None
        self._vrfs_conf = None
        self._core_service = None
        super(_CoreManager, self).__init__()

    def _run(self, *args, **kwargs):
        self._common_conf = kwargs.pop('common_conf')
        self._neighbors_conf = NeighborsConf()
        self._vrfs_conf = VrfsConf()
        from os_ken.services.protocols.bgp.core import CoreService
        self._core_service = CoreService(self._common_conf, self._neighbors_conf, self._vrfs_conf)
        waiter = kwargs.pop('waiter')
        core_activity = self._spawn_activity(self._core_service, waiter=waiter)
        core_activity.wait()

    def get_core_service(self):
        self._check_started()
        return self._core_service

    def _check_started(self):
        if not self.started:
            raise ActivityException('Cannot access any property before activity has started')

    @property
    def common_conf(self):
        self._check_started()
        return self._common_conf

    @property
    def neighbors_conf(self):
        self._check_started()
        return self._neighbors_conf

    @property
    def vrfs_conf(self):
        self._check_started()
        return self._vrfs_conf