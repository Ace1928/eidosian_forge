from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from .build import build
import copy
import json
import os
class NoEvidenceWikipediaTeacher(WikipediaTeacher):

    def __init__(self, opt, shared=None):
        self.no_evidence = True
        super().__init__(opt, shared)