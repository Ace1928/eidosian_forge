from __future__ import absolute_import, division, print_function
import base64
import time
import os
import traceback
import sys
import hashlib
from datetime import datetime
from tempfile import NamedTemporaryFile
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.hashes import (
from ansible_collections.kubernetes.core.plugins.module_utils.selector import (
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils._text import to_native, to_bytes, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.urls import Request
def _statefulset_ready(statefulset):
    updated_replicas = statefulset.status.updatedReplicas or 0
    ready_replicas = statefulset.status.readyReplicas or 0
    return statefulset.status and statefulset.spec.updateStrategy.type == 'RollingUpdate' and (statefulset.status.observedGeneration == (statefulset.metadata.generation or 0)) and (statefulset.status.updateRevision == statefulset.status.currentRevision) and (updated_replicas == statefulset.spec.replicas) and (ready_replicas == statefulset.spec.replicas) and (statefulset.status.replicas == statefulset.spec.replicas)