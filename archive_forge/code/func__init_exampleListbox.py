from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _init_exampleListbox(self, parent):
    self._exampleFrame = listframe = Frame(parent)
    self._exampleFrame.pack(fill='both', side='left', padx=2)
    self._exampleList_label = Label(self._exampleFrame, font=self._boldfont, text='Examples')
    self._exampleList_label.pack()
    self._exampleList = Listbox(self._exampleFrame, selectmode='single', relief='groove', background='white', foreground='#909090', font=self._font, selectforeground='#004040', selectbackground='#c0f0c0')
    self._exampleList.pack(side='right', fill='both', expand=1)
    for example in self._examples:
        self._exampleList.insert('end', '  %s' % example)
    self._exampleList.config(height=min(len(self._examples), 25), width=40)
    if len(self._examples) > 25:
        listscroll = Scrollbar(self._exampleFrame, orient='vertical')
        self._exampleList.config(yscrollcommand=listscroll.set)
        listscroll.config(command=self._exampleList.yview)
        listscroll.pack(side='left', fill='y')
    self._exampleList.bind('<<ListboxSelect>>', self._exampleList_select)