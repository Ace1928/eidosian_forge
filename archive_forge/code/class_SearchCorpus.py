import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
class SearchCorpus(threading.Thread):

    def __init__(self, model, page, count):
        self.model, self.count, self.page = (model, count, page)
        threading.Thread.__init__(self)

    def run(self):
        q = self.processed_query()
        sent_pos, i, sent_count = ([], 0, 0)
        for sent in self.model.tagged_sents[self.model.last_sent_searched:]:
            try:
                m = re.search(q, sent)
            except re.error:
                self.model.reset_results()
                self.model.queue.put(SEARCH_ERROR_EVENT)
                return
            if m:
                sent_pos.append((sent, m.start(), m.end()))
                i += 1
                if i > self.count:
                    self.model.last_sent_searched += sent_count - 1
                    break
            sent_count += 1
        if self.count >= len(sent_pos):
            self.model.last_sent_searched += sent_count - 1
            self.model.last_page = self.page
            self.model.set_results(self.page, sent_pos)
        else:
            self.model.set_results(self.page, sent_pos[:-1])
        self.model.queue.put(SEARCH_TERMINATED_EVENT)

    def processed_query(self):
        new = []
        for term in self.model.query.split():
            term = re.sub('\\.', '[^/ ]', term)
            if re.match('[A-Z]+$', term):
                new.append(BOUNDARY + WORD_OR_TAG + '/' + term + BOUNDARY)
            elif '/' in term:
                new.append(BOUNDARY + term + BOUNDARY)
            else:
                new.append(BOUNDARY + term + '/' + WORD_OR_TAG + BOUNDARY)
        return ' '.join(new)