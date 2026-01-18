from __future__ import absolute_import
from __future__ import print_function
from collections import namedtuple
import copy
import hashlib
import os
import six
def Override(self, layers=None, entrypoint=None, cmd=None, user=None, labels=None, env=None, ports=None, volumes=None, workdir=None, author=None, created_by=None, creation_time=None):
    return Overrides(layers=layers or self.layers, entrypoint=entrypoint or self.entrypoint, cmd=cmd or self.cmd, user=user or self.user, labels=labels or self.labels, env=env or self.env, ports=ports or self.ports, volumes=volumes or self.volumes, workdir=workdir or self.workdir, author=author or self.author, created_by=created_by or self.created_by, creation_time=creation_time or self.creation_time)