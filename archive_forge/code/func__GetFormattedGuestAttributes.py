from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import zlib
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_projector
def _GetFormattedGuestAttributes(self, guest_attributes):
    guest_attributes_json = resource_projector.MakeSerializable(guest_attributes)
    formatted_guest_attributes = {}
    for guest_attribute in guest_attributes_json:
        guest_attribute_key = guest_attribute['key']
        if guest_attribute_key in self._GUEST_ATTRIBUTES_PACKAGE_FIELD_KEYS:
            formatted_packages_info = {}
            guest_attribute_json = json.loads(guest_attribute['value'])
            for package_manager, package_list in guest_attribute_json.items():
                if package_manager in self._SPECIAL_PACKAGE_MANAGERS:
                    formatted_packages_info[package_manager] = package_list
                else:
                    formatted_packages_list = []
                    for package in package_list:
                        name = package['Name']
                        info = {'Arch': package['Arch'], 'Version': package['Version']}
                        formatted_packages_list.append({'Name': name, name: info})
                    formatted_packages_info[package_manager] = formatted_packages_list
            guest_attribute['value'] = formatted_packages_info
        formatted_guest_attributes[guest_attribute_key] = guest_attribute['value']
    return json.loads(json.dumps(formatted_guest_attributes))