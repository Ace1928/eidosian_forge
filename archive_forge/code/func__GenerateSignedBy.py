from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import flattened_printer as fp
def _GenerateSignedBy(signatures):
    sig = ', '.join((sig.keyid for sig in signatures))
    if sig == 'projects/goog-analysis/locations/global/keyRings/sbomAttestor/cryptoKeys/generatedByArtifactAnalysis/cryptoKeyVersions/1':
        return 'Artifact Analysis'
    if sig == 'projects/goog-analysis-dev/locations/global/keyRings/sbomAttestor/cryptoKeys/generatedByArtifactAnalysis/cryptoKeyVersions/1':
        return 'Artifact Analysis Dev'
    return sig