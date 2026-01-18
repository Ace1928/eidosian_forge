import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
class RoundTripModelTest(RoundTripTestCase, AllTestsBase):

    def assertRoundTripEquivalence(self, prov_doc, msg=None):
        """Exercises prov.model without the actual serialization and PROV-N
        generation.
        """
        provn_content = prov_doc.get_provn()
        self.assertEqual(prov_doc, prov_doc, 'The document is not self-equal:\n' + provn_content)