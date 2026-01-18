from itertools import chain
from nltk.internals import Counter
class FStructure(dict):

    def safeappend(self, key, item):
        """
        Append 'item' to the list at 'key'.  If no list exists for 'key', then
        construct one.
        """
        if key not in self:
            self[key] = []
        self[key].append(item)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key.lower(), value)

    def __getitem__(self, key):
        return dict.__getitem__(self, key.lower())

    def __contains__(self, key):
        return dict.__contains__(self, key.lower())

    def to_glueformula_list(self, glue_dict):
        depgraph = self.to_depgraph()
        return glue_dict.to_glueformula_list(depgraph)

    def to_depgraph(self, rel=None):
        from nltk.parse.dependencygraph import DependencyGraph
        depgraph = DependencyGraph()
        nodes = depgraph.nodes
        self._to_depgraph(nodes, 0, 'ROOT')
        for address, node in nodes.items():
            for n2 in (n for n in nodes.values() if n['rel'] != 'TOP'):
                if n2['head'] == address:
                    relation = n2['rel']
                    node['deps'].setdefault(relation, [])
                    node['deps'][relation].append(n2['address'])
        depgraph.root = nodes[1]
        return depgraph

    def _to_depgraph(self, nodes, head, rel):
        index = len(nodes)
        nodes[index].update({'address': index, 'word': self.pred[0], 'tag': self.pred[1], 'head': head, 'rel': rel})
        for feature in sorted(self):
            for item in sorted(self[feature]):
                if isinstance(item, FStructure):
                    item._to_depgraph(nodes, index, feature)
                elif isinstance(item, tuple):
                    new_index = len(nodes)
                    nodes[new_index].update({'address': new_index, 'word': item[0], 'tag': item[1], 'head': index, 'rel': feature})
                elif isinstance(item, list):
                    for n in item:
                        n._to_depgraph(nodes, index, feature)
                else:
                    raise Exception('feature %s is not an FStruct, a list, or a tuple' % feature)

    @staticmethod
    def read_depgraph(depgraph):
        return FStructure._read_depgraph(depgraph.root, depgraph)

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

    @staticmethod
    def _make_label(value):
        """
        Pick an alphabetic character as identifier for an entity in the model.

        :param value: where to index into the list of characters
        :type value: int
        """
        letter = ['f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e'][value - 1]
        num = int(value) // 26
        if num > 0:
            return letter + str(num)
        else:
            return letter

    def __repr__(self):
        return self.__str__().replace('\n', '')

    def __str__(self):
        return self.pretty_format()

    def pretty_format(self, indent=3):
        try:
            accum = '%s:[' % self.label
        except NameError:
            accum = '['
        try:
            accum += "pred '%s'" % self.pred[0]
        except NameError:
            pass
        for feature in sorted(self):
            for item in self[feature]:
                if isinstance(item, FStructure):
                    next_indent = indent + len(feature) + 3 + len(self.label)
                    accum += '\n{}{} {}'.format(' ' * indent, feature, item.pretty_format(next_indent))
                elif isinstance(item, tuple):
                    accum += "\n{}{} '{}'".format(' ' * indent, feature, item[0])
                elif isinstance(item, list):
                    accum += '\n{}{} {{{}}}'.format(' ' * indent, feature, ('\n%s' % (' ' * (indent + len(feature) + 2))).join(item))
                else:
                    raise Exception('feature %s is not an FStruct, a list, or a tuple' % feature)
        return accum + ']'