from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
class SymbolWidget(TextWidget):
    """
    A canvas widget that displays special symbols, such as the
    negation sign and the exists operator.  Symbols are specified by
    name.  Currently, the following symbol names are defined: ``neg``,
    ``disj``, ``conj``, ``lambda``, ``merge``, ``forall``, ``exists``,
    ``subseteq``, ``subset``, ``notsubset``, ``emptyset``, ``imp``,
    ``rightarrow``, ``equal``, ``notequal``, ``epsilon``.

    Attributes:

    - ``color``: the color of the text.
    - ``draggable``: whether the text can be dragged by the user.

    :cvar SYMBOLS: A dictionary mapping from symbols to the character
        in the ``symbol`` font used to render them.
    """
    SYMBOLS = {'neg': 'Ø', 'disj': 'Ú', 'conj': 'Ù', 'lambda': 'l', 'merge': 'Ä', 'forall': '"', 'exists': '$', 'subseteq': 'Í', 'subset': 'Ì', 'notsubset': 'Ë', 'emptyset': 'Æ', 'imp': 'Þ', 'rightarrow': chr(222), 'equal': '=', 'notequal': '¹', 'intersection': 'Ç', 'union': 'È', 'epsilon': 'e'}

    def __init__(self, canvas, symbol, **attribs):
        """
        Create a new symbol widget.

        :type canvas: Tkinter.Canvas
        :param canvas: This canvas widget's canvas.
        :type symbol: str
        :param symbol: The name of the symbol to display.
        :param attribs: The new canvas widget's attributes.
        """
        attribs['font'] = 'symbol'
        TextWidget.__init__(self, canvas, '', **attribs)
        self.set_symbol(symbol)

    def symbol(self):
        """
        :return: the name of the symbol that is displayed by this
            symbol widget.
        :rtype: str
        """
        return self._symbol

    def set_symbol(self, symbol):
        """
        Change the symbol that is displayed by this symbol widget.

        :type symbol: str
        :param symbol: The name of the symbol to display.
        """
        if symbol not in SymbolWidget.SYMBOLS:
            raise ValueError('Unknown symbol: %s' % symbol)
        self._symbol = symbol
        self.set_text(SymbolWidget.SYMBOLS[symbol])

    def __repr__(self):
        return '[Symbol: %r]' % self._symbol

    @staticmethod
    def symbolsheet(size=20):
        """
        Open a new Tkinter window that displays the entire alphabet
        for the symbol font.  This is useful for constructing the
        ``SymbolWidget.SYMBOLS`` dictionary.
        """
        top = Tk()

        def destroy(e, top=top):
            top.destroy()
        top.bind('q', destroy)
        Button(top, text='Quit', command=top.destroy).pack(side='bottom')
        text = Text(top, font=('helvetica', -size), width=20, height=30)
        text.pack(side='left')
        sb = Scrollbar(top, command=text.yview)
        text['yscrollcommand'] = sb.set
        sb.pack(side='right', fill='y')
        text.tag_config('symbol', font=('symbol', -size))
        for i in range(256):
            if i in (0, 10):
                continue
            for k, v in list(SymbolWidget.SYMBOLS.items()):
                if v == chr(i):
                    text.insert('end', '%-10s\t' % k)
                    break
            else:
                text.insert('end', '%-10d  \t' % i)
            text.insert('end', '[%s]\n' % chr(i), 'symbol')
        top.mainloop()