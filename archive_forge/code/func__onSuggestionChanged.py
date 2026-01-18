import gtk
def _onSuggestionChanged(self, widget, *args):
    selection = self.suggestion_list_view.get_selection()
    model, iter = selection.get_selected()
    if iter:
        suggestion = model.get_value(iter, COLUMN_SUGGESTION)
        self.replace_text.set_text(suggestion)