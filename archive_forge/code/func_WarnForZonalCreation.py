from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.core.console import console_io
def WarnForZonalCreation(self, resource_refs):
    """Warns the user if a zone has upcoming deprecation."""
    zones = self.GetZones(resource_refs)
    if not zones:
        return
    prompts = []
    zones_with_deprecated = []
    for zone in zones:
        if zone.deprecated:
            zones_with_deprecated.append(zone)
    if not zones_with_deprecated:
        return
    if zones_with_deprecated:
        phrases = []
        if len(zones_with_deprecated) == 1:
            phrases = ('zone is', 'this zone', 'the')
        else:
            phrases = ('zones are', 'these zones', 'their')
        title = '\nWARNING: The following selected {0} deprecated. All resources in {1} will be deleted after {2} turndown date.'.format(phrases[0], phrases[1], phrases[2])
        printable_deprecated_zones = []
        for zone in zones_with_deprecated:
            if zone.deprecated.deleted:
                printable_deprecated_zones.append('[{0}] {1}'.format(zone.name, zone.deprecated.deleted))
            else:
                printable_deprecated_zones.append('[{0}]'.format(zone.name))
        prompts.append(utils.ConstructList(title, printable_deprecated_zones))
    final_message = ' '.join(prompts)
    if not console_io.PromptContinue(message=final_message):
        raise exceptions.AbortedError('Creation aborted by user.')