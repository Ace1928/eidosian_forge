from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCertificateMapEntryAndCertificatesResourceArgs(parser, entry_verb, entry_noun=None, cert_verb=None, cert_noun=None, cert_group=None):
    """Add a resource argument for a Certificate Manager certificate map entry and certificates.

  NOTE: Must be used only if these are the only resource args in the command.

  Args:
    parser: the parser for the command.
    entry_verb: str, the verb to describe the entry, such as 'to update'.
    entry_noun: str, the entry resource; default: 'The certificate map entry'.
    cert_verb: str, the verb to describe the cert, default: 'to be attached to
      the entry'.
    cert_noun: str, the certificate resources; default: 'The certificates'.
    cert_group: args group certificates should belong to.
  """
    entry_noun = entry_noun or 'The certificate map entry'
    cert_noun = cert_noun or 'The certificates'
    cert_verb = cert_verb or 'to be attached to the entry'
    concept_parsers.ConceptParser([_GetCertificateMapEntryResourcePresentationSpec('entry', entry_noun, entry_verb), _GetCertificateResourcePresentationSpec('--certificates', cert_noun, cert_verb, required=False, plural=True, group=cert_group, with_location=False)]).AddToParser(parser)