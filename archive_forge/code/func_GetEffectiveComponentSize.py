from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
def GetEffectiveComponentSize(self, component_id, platform_filter=None):
    """Computes the effective size of the given component.

    If the component does not exist or does not exist on this platform, the size
    is 0.

    If it has data, just use the reported size of its data.

    If there is no data, report the total size of all its direct hidden
    dependencies (that are valid on this platform).  We don't include visible
    dependencies because they will show up in the list with their own size.

    This is a best effort estimation.  It is not easily possible to accurately
    report size in all situations because complex dependency graphs (between
    hidden and visible components, as well as circular dependencies) makes it
    infeasible to correctly show size when only displaying visible components.
    The goal is mainly to not show some components as having no size at all
    when they are wrappers around platform specific components.

    Args:
      component_id: str, The component to get the size for.
      platform_filter: platforms.Platform, A platform that components must
        match in order to be considered for any operations.

    Returns:
      int, The effective size of the component.
    """
    size = 0
    component = self.ComponentFromId(component_id)
    if component and component.platform.Matches(platform_filter):
        if component.data:
            return component.data.size
        deps = [self.ComponentFromId(d) for d in self.__dependencies[component_id]]
        deps = [d for d in deps if d.platform.Matches(platform_filter) and d.is_hidden and d.data]
        for d in deps:
            size += d.data.size or 0
    return size