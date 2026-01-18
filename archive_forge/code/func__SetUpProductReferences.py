import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _SetUpProductReferences(self, other_pbxproject, product_group, project_ref):
    for target in other_pbxproject._properties['targets']:
        if not isinstance(target, PBXNativeTarget):
            continue
        other_fileref = target._properties['productReference']
        if product_group.GetChildByRemoteObject(other_fileref) is None:
            container_item = PBXContainerItemProxy({'containerPortal': project_ref, 'proxyType': 2, 'remoteGlobalIDString': other_fileref, 'remoteInfo': target.Name()})
            reference_proxy = PBXReferenceProxy({'fileType': other_fileref._properties['explicitFileType'], 'path': other_fileref._properties['path'], 'sourceTree': other_fileref._properties['sourceTree'], 'remoteRef': container_item})
            product_group.AppendChild(reference_proxy)