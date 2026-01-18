from itertools import chain
from nltk.internals import Counter
@staticmethod
def _read_depgraph(node, depgraph, label_counter=None, parent=None):
    if not label_counter:
        label_counter = Counter()
    if node['rel'].lower() in ['spec', 'punct']:
        return (node['word'], node['tag'])
    else:
        fstruct = FStructure()
        fstruct.pred = None
        fstruct.label = FStructure._make_label(label_counter.get())
        fstruct.parent = parent
        word, tag = (node['word'], node['tag'])
        if tag[:2] == 'VB':
            if tag[2:3] == 'D':
                fstruct.safeappend('tense', ('PAST', 'tense'))
            fstruct.pred = (word, tag[:2])
        if not fstruct.pred:
            fstruct.pred = (word, tag)
        children = [depgraph.nodes[idx] for idx in chain.from_iterable(node['deps'].values())]
        for child in children:
            fstruct.safeappend(child['rel'], FStructure._read_depgraph(child, depgraph, label_counter, fstruct))
        return fstruct