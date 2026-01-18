from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
class RTEPair:
    """
    Container for RTE text-hypothesis pairs.

    The entailment relation is signalled by the ``value`` attribute in RTE1, and by
    ``entailment`` in RTE2 and RTE3. These both get mapped on to the ``entailment``
    attribute of this class.
    """

    def __init__(self, pair, challenge=None, id=None, text=None, hyp=None, value=None, task=None, length=None):
        """
        :param challenge: version of the RTE challenge (i.e., RTE1, RTE2 or RTE3)
        :param id: identifier for the pair
        :param text: the text component of the pair
        :param hyp: the hypothesis component of the pair
        :param value: classification label for the pair
        :param task: attribute for the particular NLP task that the data was drawn from
        :param length: attribute for the length of the text of the pair
        """
        self.challenge = challenge
        self.id = pair.attrib['id']
        self.gid = f'{self.challenge}-{self.id}'
        self.text = pair[0].text
        self.hyp = pair[1].text
        if 'value' in pair.attrib:
            self.value = norm(pair.attrib['value'])
        elif 'entailment' in pair.attrib:
            self.value = norm(pair.attrib['entailment'])
        else:
            self.value = value
        if 'task' in pair.attrib:
            self.task = pair.attrib['task']
        else:
            self.task = task
        if 'length' in pair.attrib:
            self.length = pair.attrib['length']
        else:
            self.length = length

    def __repr__(self):
        if self.challenge:
            return f'<RTEPair: gid={self.challenge}-{self.id}>'
        else:
            return '<RTEPair: id=%s>' % self.id