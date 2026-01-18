import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def _get_pending_jobs_affecting_element(self, element):
    mappings = self._conn.Msvm_AffectedJobElement(AffectedElement=element.path_())
    pending_jobs = []
    for mapping in mappings:
        try:
            if mapping.AffectingElement and (not self._is_job_completed(mapping.AffectingElement)):
                pending_jobs.append(mapping.AffectingElement)
        except exceptions.x_wmi as ex:
            if not _utils._is_not_found_exc(ex):
                raise
    return pending_jobs