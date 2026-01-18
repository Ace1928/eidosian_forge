from reportlab.platypus.flowables import *
from reportlab.platypus.flowables import _ContainerSpace
from reportlab.lib.units import inch
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.frames import Frame
from reportlab.rl_config import defaultPageSize, verbose
import reportlab.lib.sequencer
from reportlab.pdfgen import canvas
from reportlab.lib.utils import isSeq, encode_label, decode_label, annotateException, strTypes
import sys
import logging
class IndexingFlowable(Flowable):
    """Abstract interface definition for flowables which might
    hold references to other pages or themselves be targets
    of cross-references.  XRefStart, XRefDest, Table of Contents,
    Indexes etc."""

    def isIndexing(self):
        return 1

    def isSatisfied(self):
        return 1

    def notify(self, kind, stuff):
        """This will be called by the framework wherever 'stuff' happens.
        'kind' will be a value that can be used to decide whether to
        pay attention or not."""
        pass

    def beforeBuild(self):
        """Called by multiBuild before it starts; use this to clear
        old contents"""
        pass

    def afterBuild(self):
        """Called after build ends but before isSatisfied"""
        pass