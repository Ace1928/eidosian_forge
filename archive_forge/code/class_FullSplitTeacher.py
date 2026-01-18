from parlai.core.teachers import DialogTeacher, ChunkTeacher, ChunkOutput
from parlai.core.message import Message
from .build import build
import json
import os
from typing import List, Tuple
class FullSplitTeacher(ChunkTeacher):
    """
    Full Wikipedia teacher that splits the chunks into train/valid/test.
    """

    def __init__(self, opt, shared=None):
        self.TRAINSIZE = 5437097
        self.VALIDSIZE = 71052
        self.TESTSIZE = 39975
        if shared is None:
            self.opt = opt
            self._set_chunk_idx_to_file()
        else:
            self.chunk_idx_to_file = shared['chunk_idx_to_file']
        super().__init__(opt, shared)

    def _get_data_folder(self):
        return os.path.join(self.opt['datapath'], 'wikipedia/full/wiki_full_extracted')

    def get_num_samples(self, opt) -> Tuple[int, int]:
        """
        Return the number of samples given the datatype.
        """
        datatype = opt['datatype']
        if 'train' in datatype:
            return (self.TRAINSIZE, self.TRAINSIZE)
        elif 'valid' in datatype:
            return (self.VALIDSIZE, self.VALIDSIZE)
        else:
            return (self.TESTSIZE, self.TESTSIZE)

    def _set_chunk_idx_to_file(self):
        folder = self._get_data_folder()
        all_subdirs = sorted([x for x in os.listdir(folder) if 'README' not in x])
        self.chunk_idx_to_file = {i: x for i, x in enumerate(all_subdirs)}

    def get_fold_chunks(self, opt) -> List[int]:
        """
        Return a list of chunk IDs (integer).

        Given the datatype (train/test/valid), return the list of chunk IDs that
        correspond to that split.
        """
        datatype = opt['datatype']
        all_chunk_idxs = list(self.chunk_idx_to_file.keys())
        if 'train' in datatype:
            return all_chunk_idxs[:-2]
        elif 'valid' in datatype:
            return [all_chunk_idxs[-2]]
        else:
            return [all_chunk_idxs[-1]]

    def load_from_chunk(self, chunk_idx: int):
        """
        Given the chunk index, load examples from that chunk.

        Return a list of tuples. The function `_create_message` will take these tuples
        to form the Message object that is returned by the teacher.
        """
        output = []
        chunk_path = os.path.join(self.folder, self.chunk_idx_to_file[chunk_idx])
        for wiki_file in os.listdir(chunk_path):
            wiki_file_path = os.path.join(chunk_path, wiki_file)
            with open(wiki_file_path) as wf:
                for article_json in wf:
                    article = json.loads(article_json)
                    title = article['title']
                    text = article['text']
                    output.append((title, text))
        return output

    def create_message(self, queue_output: ChunkOutput, entry_idx=0) -> 'Message':
        """
        Given the tuple output of the queue, return an act.
        """
        title, text = queue_output
        return Message({'title': title, 'text': text, 'labels': [''], 'episode_done': True})

    def share(self):
        shared = super().share()
        shared['chunk_idx_to_file'] = self.chunk_idx_to_file
        return shared