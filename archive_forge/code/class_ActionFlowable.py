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
class ActionFlowable(Flowable):
    """This Flowable is never drawn, it can be used for data driven controls
       For example to change a page template (from one column to two, for example)
       use NextPageTemplate which creates an ActionFlowable.
    """

    def __init__(self, action=()):
        Flowable.__init__(self)
        if not isSeq(action):
            action = (action,)
        self.action = tuple(action)

    def apply(self, doc):
        """
        This is called by the doc.build processing to allow the instance to
        implement its behaviour
        """
        action = self.action[0]
        args = tuple(self.action[1:])
        arn = 'handle_' + action
        if arn == 'handle_nextPageTemplate' and args[0] == 'main':
            pass
        try:
            getattr(doc, arn)(*args)
        except AttributeError as aerr:
            if aerr.args[0] == arn:
                raise NotImplementedError("Can't handle ActionFlowable(%s)" % action)
            else:
                raise
        except:
            annotateException('\nhandle_%s args=%s' % (action, ascii(args)))

    def __call__(self):
        return self

    def identity(self, maxLen=None):
        return 'ActionFlowable: %s%s' % (str(self.action), self._frameName())