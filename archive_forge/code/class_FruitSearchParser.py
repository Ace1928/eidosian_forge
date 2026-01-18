from searchparser import SearchQueryParser
class FruitSearchParser(SearchQueryParser):

    def GetWord(self, word):
        return {p for p in products if p.startswith(word + ' ')}

    def GetWordWildcard(self, word):
        return {p for p in products if p.startswith(word[:-1])}

    def GetQuotes(self, search_string, tmp_result):
        result = set()
        return result

    def GetNot(self, not_set):
        return set(products) - not_set