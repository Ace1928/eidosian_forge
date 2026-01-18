from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _UpdateCertificateTemplateFromArgs(self, args, current_labels):
    """Creates a Certificate template object and update mask from Certificate template update flags.

    Requires that args has 'description', 'copy-sans', 'copy-subject',
    'predefined-values-file', 'copy-known-extensions', 'copy-extensions-by-oid',
    and update labels flags registered.

    Args:
      args: The parser that contains the flag values.
      current_labels: The current set of labels for the Certificate Template.

    Returns:
      A tuple with the Certificate template object to update with and the list
      of
      strings representing the update mask, respectively.
    """
    messages = privateca_base.GetMessagesModule('v1')
    template_to_update = messages.CertificateTemplate()
    update_mask = []
    if args.IsSpecified('copy_sans') or args.IsSpecified('copy_subject') or args.IsSpecified('identity_cel_expression'):
        template_to_update.identityConstraints = flags.ParseIdentityConstraints(args)
        if args.IsSpecified('copy_sans'):
            update_mask.append('identity_constraints.allow_subject_alt_names_passthrough')
        if args.IsSpecified('copy_subject'):
            update_mask.append('identity_constraints.allow_subject_passthrough')
        if args.IsSpecified('identity_cel_expression'):
            update_mask.append('identity_constraints.cel_expression')
    if args.IsSpecified('predefined_values_file'):
        template_to_update.predefinedValues = flags.ParsePredefinedValues(args)
        update_mask.append('predefined_values')
    if args.IsSpecified('description'):
        template_to_update.description = args.description
        update_mask.append('description')
    known_exts_flags = args.IsSpecified('copy_known_extensions') or args.IsSpecified('drop_known_extensions')
    oid_exts_flags = args.IsSpecified('copy_extensions_by_oid') or args.IsSpecified('drop_oid_extensions')
    if known_exts_flags or oid_exts_flags:
        template_to_update.passthroughExtensions = flags.ParseExtensionConstraints(args)
        if known_exts_flags:
            update_mask.append('passthrough_extensions.known_extensions')
        if oid_exts_flags:
            update_mask.append('passthrough_extensions.additional_extensions')
    labels_diff = labels_util.Diff.FromUpdateArgs(args)
    labels_update = labels_diff.Apply(messages.CaPool.LabelsValue, current_labels)
    if labels_update.needs_update:
        template_to_update.labels = labels_update.labels
        update_mask.append('labels')
    if not update_mask:
        raise privateca_exceptions.NoUpdateException('No updates found for the requested certificate template.')
    return (template_to_update, update_mask)