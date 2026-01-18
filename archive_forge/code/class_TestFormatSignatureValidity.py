import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestFormatSignatureValidity(tests.TestCaseWithTransport):

    def verify_revision_signature(self, revid, gpg_strategy):
        return (gpg.SIGNATURE_VALID, 'UTF8 Test ¡±ÁÑáñ <jrandom@example.com>')

    def test_format_signature_validity_utf(self):
        """Check that GPG signatures containing UTF-8 names are formatted
        correctly."""
        wt = self.make_branch_and_tree('.')
        revid = wt.commit('empty commit')
        repo = wt.branch.repository
        self.overrideAttr(repo, 'verify_revision_signature', self.verify_revision_signature)
        out = log.format_signature_validity(revid, wt.branch)
        self.assertEqual('valid signature from UTF8 Test ¡±ÁÑáñ <jrandom@example.com>', out)