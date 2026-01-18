from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
class PPAttachmentCorpusReader(CorpusReader):
    """
    sentence_id verb noun1 preposition noun2 attachment
    """

    def attachments(self, fileids):
        return concat([StreamBackedCorpusView(fileid, self._read_obj_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def tuples(self, fileids):
        return concat([StreamBackedCorpusView(fileid, self._read_tuple_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def _read_tuple_block(self, stream):
        line = stream.readline()
        if line:
            return [tuple(line.split())]
        else:
            return []

    def _read_obj_block(self, stream):
        line = stream.readline()
        if line:
            return [PPAttachment(*line.split())]
        else:
            return []