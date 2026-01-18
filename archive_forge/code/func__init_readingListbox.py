from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _init_readingListbox(self, parent):
    self._readingFrame = listframe = Frame(parent)
    self._readingFrame.pack(fill='both', side='left', padx=2)
    self._readingList_label = Label(self._readingFrame, font=self._boldfont, text='Readings')
    self._readingList_label.pack()
    self._readingList = Listbox(self._readingFrame, selectmode='single', relief='groove', background='white', foreground='#909090', font=self._font, selectforeground='#004040', selectbackground='#c0f0c0')
    self._readingList.pack(side='right', fill='both', expand=1)
    listscroll = Scrollbar(self._readingFrame, orient='vertical')
    self._readingList.config(yscrollcommand=listscroll.set)
    listscroll.config(command=self._readingList.yview)
    listscroll.pack(side='right', fill='y')
    self._populate_readingListbox()