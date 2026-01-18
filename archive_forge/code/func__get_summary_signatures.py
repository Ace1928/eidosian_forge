import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _get_summary_signatures(self):
    """Verifies and returns the summary signatures.

    Returns:
      A dictionary of the signature identifiers {signature: index} that will be
      computed when trace_mode is summary.
    """
    signatures = self._flag_value_as_list(FLAG_NAME_SUMMARY_SIGNATURES)
    supported_signatures = self._supported_signatures()
    tt_signatures = []
    for signature in signatures:
        signature_with_prefix = '%s_%s' % (_TT_PREFIX, signature)
        if signature in supported_signatures:
            tt_signatures.append(signature)
        elif signature_with_prefix in supported_signatures:
            tt_signatures.append(signature_with_prefix)
        else:
            logging.warning('Unknown signature:%s. Supported signatures: %s' % (signature, supported_signatures))
    if not tt_signatures:
        return {TT_SUMMARY_MAX_ABS: 0, TT_SUMMARY_NORM: 1}
    else:
        return {signature: idx for idx, signature in enumerate(tt_signatures)}