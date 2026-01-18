import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
class TransitionParser(ParserI):
    """
    Class for transition based parser. Implement 2 algorithms which are "arc-standard" and "arc-eager"
    """
    ARC_STANDARD = 'arc-standard'
    ARC_EAGER = 'arc-eager'

    def __init__(self, algorithm):
        """
        :param algorithm: the algorithm option of this parser. Currently support `arc-standard` and `arc-eager` algorithm
        :type algorithm: str
        """
        if not algorithm in [self.ARC_STANDARD, self.ARC_EAGER]:
            raise ValueError(' Currently we only support %s and %s ' % (self.ARC_STANDARD, self.ARC_EAGER))
        self._algorithm = algorithm
        self._dictionary = {}
        self._transition = {}
        self._match_transition = {}

    def _get_dep_relation(self, idx_parent, idx_child, depgraph):
        p_node = depgraph.nodes[idx_parent]
        c_node = depgraph.nodes[idx_child]
        if c_node['word'] is None:
            return None
        if c_node['head'] == p_node['address']:
            return c_node['rel']
        else:
            return None

    def _convert_to_binary_features(self, features):
        """
        :param features: list of feature string which is needed to convert to binary features
        :type features: list(str)
        :return : string of binary features in libsvm format  which is 'featureID:value' pairs
        """
        unsorted_result = []
        for feature in features:
            self._dictionary.setdefault(feature, len(self._dictionary))
            unsorted_result.append(self._dictionary[feature])
        return ' '.join((str(featureID) + ':1.0' for featureID in sorted(unsorted_result)))

    def _is_projective(self, depgraph):
        arc_list = []
        for key in depgraph.nodes:
            node = depgraph.nodes[key]
            if 'head' in node:
                childIdx = node['address']
                parentIdx = node['head']
                if parentIdx is not None:
                    arc_list.append((parentIdx, childIdx))
        for parentIdx, childIdx in arc_list:
            if childIdx > parentIdx:
                temp = childIdx
                childIdx = parentIdx
                parentIdx = temp
            for k in range(childIdx + 1, parentIdx):
                for m in range(len(depgraph.nodes)):
                    if m < childIdx or m > parentIdx:
                        if (k, m) in arc_list:
                            return False
                        if (m, k) in arc_list:
                            return False
        return True

    def _write_to_file(self, key, binary_features, input_file):
        """
        write the binary features to input file and update the transition dictionary
        """
        self._transition.setdefault(key, len(self._transition) + 1)
        self._match_transition[self._transition[key]] = key
        input_str = str(self._transition[key]) + ' ' + binary_features + '\n'
        input_file.write(input_str.encode('utf-8'))

    def _create_training_examples_arc_std(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : Page 32, Chapter 3. Dependency Parsing by Sandra Kubler, Ryan McDonal and Joakim Nivre (2009)
        """
        operation = Transition(self.ARC_STANDARD)
        count_proj = 0
        training_seq = []
        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue
            count_proj += 1
            conf = Configuration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)
                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = Transition.LEFT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        precondition = True
                        maxID = conf._max_address
                        for w in range(maxID + 1):
                            if w != b0:
                                relw = self._get_dep_relation(b0, w, depgraph)
                                if relw is not None:
                                    if (b0, relw, w) not in conf.arcs:
                                        precondition = False
                        if precondition:
                            key = Transition.RIGHT_ARC + ':' + rel
                            self._write_to_file(key, binary_features, input_file)
                            operation.right_arc(conf, rel)
                            training_seq.append(key)
                            continue
                key = Transition.SHIFT
                self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)
                training_seq.append(key)
        print(' Number of training examples : ' + str(len(depgraphs)))
        print(' Number of valid (projective) examples : ' + str(count_proj))
        return training_seq

    def _create_training_examples_arc_eager(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : 'A Dynamic Oracle for Arc-Eager Dependency Parsing' by Joav Goldberg and Joakim Nivre
        """
        operation = Transition(self.ARC_EAGER)
        countProj = 0
        training_seq = []
        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue
            countProj += 1
            conf = Configuration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)
                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = Transition.LEFT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        key = Transition.RIGHT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.right_arc(conf, rel)
                        training_seq.append(key)
                        continue
                    flag = False
                    for k in range(s0):
                        if self._get_dep_relation(k, b0, depgraph) is not None:
                            flag = True
                        if self._get_dep_relation(b0, k, depgraph) is not None:
                            flag = True
                    if flag:
                        key = Transition.REDUCE
                        self._write_to_file(key, binary_features, input_file)
                        operation.reduce(conf)
                        training_seq.append(key)
                        continue
                key = Transition.SHIFT
                self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)
                training_seq.append(key)
        print(' Number of training examples : ' + str(len(depgraphs)))
        print(' Number of valid (projective) examples : ' + str(countProj))
        return training_seq

    def train(self, depgraphs, modelfile, verbose=True):
        """
        :param depgraphs : list of DependencyGraph as the training data
        :type depgraphs : DependencyGraph
        :param modelfile : file name to save the trained model
        :type modelfile : str
        """
        try:
            input_file = tempfile.NamedTemporaryFile(prefix='transition_parse.train', dir=tempfile.gettempdir(), delete=False)
            if self._algorithm == self.ARC_STANDARD:
                self._create_training_examples_arc_std(depgraphs, input_file)
            else:
                self._create_training_examples_arc_eager(depgraphs, input_file)
            input_file.close()
            x_train, y_train = load_svmlight_file(input_file.name)
            model = svm.SVC(kernel='poly', degree=2, coef0=0, gamma=0.2, C=0.5, verbose=verbose, probability=True)
            model.fit(x_train, y_train)
            pickle.dump(model, open(modelfile, 'wb'))
        finally:
            remove(input_file.name)

    def parse(self, depgraphs, modelFile):
        """
        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :param modelfile: the model file
        :type modelfile: str
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        """
        result = []
        model = pickle.load(open(modelFile, 'rb'))
        operation = Transition(self._algorithm)
        for depgraph in depgraphs:
            conf = Configuration(depgraph)
            while len(conf.buffer) > 0:
                features = conf.extract_features()
                col = []
                row = []
                data = []
                for feature in features:
                    if feature in self._dictionary:
                        col.append(self._dictionary[feature])
                        row.append(0)
                        data.append(1.0)
                np_col = array(sorted(col))
                np_row = array(row)
                np_data = array(data)
                x_test = sparse.csr_matrix((np_data, (np_row, np_col)), shape=(1, len(self._dictionary)))
                prob_dict = {}
                pred_prob = model.predict_proba(x_test)[0]
                for i in range(len(pred_prob)):
                    prob_dict[i] = pred_prob[i]
                sorted_Prob = sorted(prob_dict.items(), key=itemgetter(1), reverse=True)
                for y_pred_idx, confidence in sorted_Prob:
                    y_pred = model.classes_[y_pred_idx]
                    if y_pred in self._match_transition:
                        strTransition = self._match_transition[y_pred]
                        baseTransition = strTransition.split(':')[0]
                        if baseTransition == Transition.LEFT_ARC:
                            if operation.left_arc(conf, strTransition.split(':')[1]) != -1:
                                break
                        elif baseTransition == Transition.RIGHT_ARC:
                            if operation.right_arc(conf, strTransition.split(':')[1]) != -1:
                                break
                        elif baseTransition == Transition.REDUCE:
                            if operation.reduce(conf) != -1:
                                break
                        elif baseTransition == Transition.SHIFT:
                            if operation.shift(conf) != -1:
                                break
                    else:
                        raise ValueError('The predicted transition is not recognized, expected errors')
            new_depgraph = deepcopy(depgraph)
            for key in new_depgraph.nodes:
                node = new_depgraph.nodes[key]
                node['rel'] = ''
                node['head'] = 0
            for head, rel, child in conf.arcs:
                c_node = new_depgraph.nodes[child]
                c_node['head'] = head
                c_node['rel'] = rel
            result.append(new_depgraph)
        return result