from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from .build import build
import copy
import os
class EchoTeacher(FunpediaTeacher):
    """
    Replaces answers with an echo of the passage.

    Useful for measuring how much a model learns to simply repeat what is said.
    """

    def _setup_data(self):
        super()._setup_data()
        for i in range(len(self.entries)):
            self.entries[i]['label'] = self.entries[i]['passage']