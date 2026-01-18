import gtk
class GtkSpellCheckerDialog(gtk.Window):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_title('Spell check')
        self.set_default_size(350, 200)
        self._checker = None
        self._numContext = 40
        self.errors = None
        accel_group = gtk.AccelGroup()
        self.add_accel_group(accel_group)
        self._conditional_widgets = []
        conditional = self._conditional_widgets.append
        mainbox = gtk.VBox(spacing=5)
        hbox = gtk.HBox(spacing=5)
        self.add(mainbox)
        mainbox.pack_start(hbox, padding=5)
        box1 = gtk.VBox(spacing=5)
        hbox.pack_start(box1, padding=5)
        conditional(box1)
        text_view_lable = gtk.Label('Unrecognized word')
        text_view_lable.set_justify(gtk.JUSTIFY_LEFT)
        box1.pack_start(text_view_lable, False, False)
        text_view = gtk.TextView()
        text_view.set_wrap_mode(gtk.WRAP_WORD)
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        self.error_text = text_view.get_buffer()
        text_buffer = text_view.get_buffer()
        text_buffer.create_tag('fg_black', foreground='black')
        text_buffer.create_tag('fg_red', foreground='red')
        box1.pack_start(text_view)
        change_to_box = gtk.HBox()
        box1.pack_start(change_to_box, False, False)
        change_to_label = gtk.Label('Change to:')
        self.replace_text = gtk.Entry()
        text_view_lable.set_justify(gtk.JUSTIFY_LEFT)
        change_to_box.pack_start(change_to_label, False, False)
        change_to_box.pack_start(self.replace_text)
        sw = gtk.ScrolledWindow()
        sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        box1.pack_start(sw)
        self.suggestion_list_view = create_list_view('Suggestions')
        self.suggestion_list_view.connect('button_press_event', self._onButtonPress)
        self.suggestion_list_view.connect('cursor-changed', self._onSuggestionChanged)
        sw.add(self.suggestion_list_view)
        button_box = gtk.VButtonBox()
        hbox.pack_start(button_box, False, False)
        button = gtk.Button('Ignore')
        button.connect('clicked', self._onIgnore)
        button.add_accelerator('activate', accel_group, gtk.keysyms.Return, 0, gtk.ACCEL_VISIBLE)
        button_box.pack_start(button)
        conditional(button)
        button = gtk.Button('Ignore All')
        button.connect('clicked', self._onIgnoreAll)
        button_box.pack_start(button)
        conditional(button)
        button = gtk.Button('Replace')
        button.connect('clicked', self._onReplace)
        button_box.pack_start(button)
        conditional(button)
        button = gtk.Button('Replace All')
        button.connect('clicked', self._onReplaceAll)
        button_box.pack_start(button)
        conditional(button)
        button = gtk.Button('_Add')
        button.connect('clicked', self._onAdd)
        button_box.pack_start(button)
        conditional(button)
        button = gtk.Button(stock=gtk.STOCK_CLOSE)
        button.connect('clicked', self._onClose)
        button.add_accelerator('activate', accel_group, gtk.keysyms.Escape, 0, gtk.ACCEL_VISIBLE)
        button_box.pack_end(button)
        self._dict_lable = gtk.Label('')
        mainbox.pack_start(self._dict_lable, False, False, padding=5)
        mainbox.show_all()

    def _onIgnore(self, w, *args):
        print(['ignore'])
        self._advance()

    def _onIgnoreAll(self, w, *args):
        print(['ignore all'])
        self._checker.ignore_always()
        self._advance()

    def _onReplace(self, *args):
        print(['Replace'])
        repl = self._getRepl()
        self._checker.replace(repl)
        self._advance()

    def _onReplaceAll(self, *args):
        print(['Replace all'])
        repl = self._getRepl()
        self._checker.replace_always(repl)
        self._advance()

    def _onAdd(self, *args):
        """Callback for the "add" button."""
        self._checker.add()
        self._advance()

    def _onClose(self, w, *args):
        self.emit('delete_event', gtk.gdk.Event(gtk.gdk.BUTTON_PRESS))
        return True

    def _onButtonPress(self, widget, event):
        if event.type == gtk.gdk._2BUTTON_PRESS:
            print(['Double click!'])
            self._onReplace()

    def _onSuggestionChanged(self, widget, *args):
        selection = self.suggestion_list_view.get_selection()
        model, iter = selection.get_selected()
        if iter:
            suggestion = model.get_value(iter, COLUMN_SUGGESTION)
            self.replace_text.set_text(suggestion)

    def _getRepl(self):
        """Get the chosen replacement string."""
        repl = self.replace_text.get_text()
        repl = self._checker.coerce_string(repl)
        return repl

    def _fillSuggestionList(self, suggestions):
        model = self.suggestion_list_view.get_model()
        model.clear()
        for suggestion in suggestions:
            value = '%s' % (suggestion,)
            model.append([value])

    def setSpellChecker(self, checker):
        assert checker, "checker can't be None"
        self._checker = checker
        self._dict_lable.set_text('Dictionary:%s' % (checker.dict.tag,))

    def getSpellChecker(self, checker):
        return self._checker

    def updateUI(self):
        self._advance()

    def _disableButtons(self):
        for w in self._conditional_widgets:
            w.set_sensitive(False)

    def _enableButtons(self):
        for w in self._conditional_widgets:
            w.set_sensitive(True)

    def _advance(self):
        """Advance to the next error.
        This method advances the SpellChecker to the next error, if
        any.  It then displays the error and some surrounding context,
        and well as listing the suggested replacements.
        """
        if self._checker is None:
            self._disableButtons()
            self.emit('check-done')
            return
        try:
            self._checker.next()
        except StopIteration:
            self._disableButtons()
            self.error_text.set_text('')
            self._fillSuggestionList([])
            self.replace_text.set_text('')
            return
        self._enableButtons()
        self.error_text.set_text('')
        iter = self.error_text.get_iter_at_offset(0)
        append = self.error_text.insert_with_tags_by_name
        lContext = self._checker.leading_context(self._numContext)
        tContext = self._checker.trailing_context(self._numContext)
        append(iter, lContext, 'fg_black')
        append(iter, self._checker.word, 'fg_red')
        append(iter, tContext, 'fg_black')
        suggs = self._checker.suggest()
        self._fillSuggestionList(suggs)
        if suggs:
            self.replace_text.set_text(suggs[0])
        else:
            self.replace_text.set_text('')