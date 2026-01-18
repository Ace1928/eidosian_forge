from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
class ParserTest(SearchQueryParser):
    """Tests the parser with some search queries
    tests containts a dictionary with tests and expected results.
    """
    tests = {'help': {1, 2, 4, 5}, 'help or hulp': {1, 2, 3, 4, 5}, 'help and hulp': {2}, 'help hulp': {2}, 'help and hulp or hilp': {2, 3, 4}, 'help or hulp and hilp': {1, 2, 3, 4, 5}, 'help or hulp or hilp or halp': {1, 2, 3, 4, 5, 6}, '(help or hulp) and (hilp or halp)': {3, 4, 5}, 'help and (hilp or halp)': {4, 5}, '(help and (hilp or halp)) or hulp': {2, 3, 4, 5}, 'not help': {3, 6, 7, 8}, 'not hulp and halp': {5, 6}, 'not (help and halp)': {1, 2, 3, 4, 6, 7, 8}, '"help me please"': {2}, '"help me please" or hulp': {2, 3}, '"help me please" or (hulp and halp)': {2}, 'help*': {1, 2, 4, 5, 8}, 'help or hulp*': {1, 2, 3, 4, 5}, 'help* and hulp': {2}, 'help and hulp* or hilp': {2, 3, 4}, 'help* or hulp or hilp or halp': {1, 2, 3, 4, 5, 6, 8}, '(help or hulp*) and (hilp* or halp)': {3, 4, 5}, 'help* and (hilp* or halp*)': {4, 5}, '(help and (hilp* or halp)) or hulp*': {2, 3, 4, 5}, 'not help* and halp': {6}, 'not (help* and helpe*)': {1, 2, 3, 4, 5, 6, 7}, '"help* me please"': {2}, '"help* me* please" or hulp*': {2, 3}, '"help me please*" or (hulp and halp)': {2}, '"help me please" not (hulp and halp)': {2}, '"help me please" hulp': {2}, 'help and hilp and not holp': {4}, 'help hilp not holp': {4}, 'help hilp and not holp': {4}}
    docs = {1: 'help', 2: 'help me please hulp', 3: 'hulp hilp', 4: 'help hilp', 5: 'halp thinks he needs help', 6: 'he needs halp', 7: 'nothing', 8: 'helper'}
    index = {'help': {1, 2, 4, 5}, 'me': {2}, 'please': {2}, 'hulp': {2, 3}, 'hilp': {3, 4}, 'halp': {5, 6}, 'thinks': {5}, 'he': {5, 6}, 'needs': {5, 6}, 'nothing': {7}, 'helper': {8}}

    def GetWord(self, word):
        if word in self.index:
            return self.index[word]
        else:
            return set()

    def GetWordWildcard(self, word):
        result = set()
        for item in list(self.index.keys()):
            if word == item[0:len(word)]:
                result = result.union(self.index[item])
        return result

    def GetQuotes(self, search_string, tmp_result):
        result = set()
        for item in tmp_result:
            if self.docs[item].count(search_string):
                result.add(item)
        return result

    def GetNot(self, not_set):
        all = set(list(self.docs.keys()))
        return all.difference(not_set)

    def Test(self):
        all_ok = True
        for item in list(self.tests.keys()):
            print(item)
            r = self.Parse(item)
            e = self.tests[item]
            print('Result: %s' % r)
            print('Expect: %s' % e)
            if e == r:
                print('Test OK')
            else:
                all_ok = False
                print('>>>>>>>>>>>>>>>>>>>>>>Test ERROR<<<<<<<<<<<<<<<<<<<<<')
            print('')
        return all_ok