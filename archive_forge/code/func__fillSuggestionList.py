import gtk
def _fillSuggestionList(self, suggestions):
    model = self.suggestion_list_view.get_model()
    model.clear()
    for suggestion in suggestions:
        value = '%s' % (suggestion,)
        model.append([value])