from __future__ import annotations
import re
from typing import Any, List, Optional
from langchain_text_splitters.base import Language, TextSplitter
class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(self, separators: Optional[List[str]]=None, keep_separator: bool=True, is_separator_regex: bool=False, **kwargs: Any) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ['\n\n', '\n', ' ', '']
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == '':
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break
        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)
        _good_splits = []
        _separator = '' if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(cls, language: Language, **kwargs: Any) -> RecursiveCharacterTextSplitter:
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        if language == Language.CPP:
            return ['\nclass ', '\nvoid ', '\nint ', '\nfloat ', '\ndouble ', '\nif ', '\nfor ', '\nwhile ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.GO:
            return ['\nfunc ', '\nvar ', '\nconst ', '\ntype ', '\nif ', '\nfor ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.JAVA:
            return ['\nclass ', '\npublic ', '\nprotected ', '\nprivate ', '\nstatic ', '\nif ', '\nfor ', '\nwhile ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.KOTLIN:
            return ['\nclass ', '\npublic ', '\nprotected ', '\nprivate ', '\ninternal ', '\ncompanion ', '\nfun ', '\nval ', '\nvar ', '\nif ', '\nfor ', '\nwhile ', '\nwhen ', '\ncase ', '\nelse ', '\n\n', '\n', ' ', '']
        elif language == Language.JS:
            return ['\nfunction ', '\nconst ', '\nlet ', '\nvar ', '\nclass ', '\nif ', '\nfor ', '\nwhile ', '\nswitch ', '\ncase ', '\ndefault ', '\n\n', '\n', ' ', '']
        elif language == Language.TS:
            return ['\nenum ', '\ninterface ', '\nnamespace ', '\ntype ', '\nclass ', '\nfunction ', '\nconst ', '\nlet ', '\nvar ', '\nif ', '\nfor ', '\nwhile ', '\nswitch ', '\ncase ', '\ndefault ', '\n\n', '\n', ' ', '']
        elif language == Language.PHP:
            return ['\nfunction ', '\nclass ', '\nif ', '\nforeach ', '\nwhile ', '\ndo ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.PROTO:
            return ['\nmessage ', '\nservice ', '\nenum ', '\noption ', '\nimport ', '\nsyntax ', '\n\n', '\n', ' ', '']
        elif language == Language.PYTHON:
            return ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
        elif language == Language.RST:
            return ['\n=+\n', '\n-+\n', '\n\\*+\n', '\n\n.. *\n\n', '\n\n', '\n', ' ', '']
        elif language == Language.RUBY:
            return ['\ndef ', '\nclass ', '\nif ', '\nunless ', '\nwhile ', '\nfor ', '\ndo ', '\nbegin ', '\nrescue ', '\n\n', '\n', ' ', '']
        elif language == Language.RUST:
            return ['\nfn ', '\nconst ', '\nlet ', '\nif ', '\nwhile ', '\nfor ', '\nloop ', '\nmatch ', '\nconst ', '\n\n', '\n', ' ', '']
        elif language == Language.SCALA:
            return ['\nclass ', '\nobject ', '\ndef ', '\nval ', '\nvar ', '\nif ', '\nfor ', '\nwhile ', '\nmatch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.SWIFT:
            return ['\nfunc ', '\nclass ', '\nstruct ', '\nenum ', '\nif ', '\nfor ', '\nwhile ', '\ndo ', '\nswitch ', '\ncase ', '\n\n', '\n', ' ', '']
        elif language == Language.MARKDOWN:
            return ['\n#{1,6} ', '```\n', '\n\\*\\*\\*+\n', '\n---+\n', '\n___+\n', '\n\n', '\n', ' ', '']
        elif language == Language.LATEX:
            return ['\n\\\\chapter{', '\n\\\\section{', '\n\\\\subsection{', '\n\\\\subsubsection{', '\n\\\\begin{enumerate}', '\n\\\\begin{itemize}', '\n\\\\begin{description}', '\n\\\\begin{list}', '\n\\\\begin{quote}', '\n\\\\begin{quotation}', '\n\\\\begin{verse}', '\n\\\\begin{verbatim}', '\n\\\x08egin{align}', '$$', '$', ' ', '']
        elif language == Language.HTML:
            return ['<body', '<div', '<p', '<br', '<li', '<h1', '<h2', '<h3', '<h4', '<h5', '<h6', '<span', '<table', '<tr', '<td', '<th', '<ul', '<ol', '<header', '<footer', '<nav', '<head', '<style', '<script', '<meta', '<title', '']
        elif language == Language.CSHARP:
            return ['\ninterface ', '\nenum ', '\nimplements ', '\ndelegate ', '\nevent ', '\nclass ', '\nabstract ', '\npublic ', '\nprotected ', '\nprivate ', '\nstatic ', '\nreturn ', '\nif ', '\ncontinue ', '\nfor ', '\nforeach ', '\nwhile ', '\nswitch ', '\nbreak ', '\ncase ', '\nelse ', '\ntry ', '\nthrow ', '\nfinally ', '\ncatch ', '\n\n', '\n', ' ', '']
        elif language == Language.SOL:
            return ['\npragma ', '\nusing ', '\ncontract ', '\ninterface ', '\nlibrary ', '\nconstructor ', '\ntype ', '\nfunction ', '\nevent ', '\nmodifier ', '\nerror ', '\nstruct ', '\nenum ', '\nif ', '\nfor ', '\nwhile ', '\ndo while ', '\nassembly ', '\n\n', '\n', ' ', '']
        elif language == Language.COBOL:
            return ['\nIDENTIFICATION DIVISION.', '\nENVIRONMENT DIVISION.', '\nDATA DIVISION.', '\nPROCEDURE DIVISION.', '\nWORKING-STORAGE SECTION.', '\nLINKAGE SECTION.', '\nFILE SECTION.', '\nINPUT-OUTPUT SECTION.', '\nOPEN ', '\nCLOSE ', '\nREAD ', '\nWRITE ', '\nIF ', '\nELSE ', '\nMOVE ', '\nPERFORM ', '\nUNTIL ', '\nVARYING ', '\nACCEPT ', '\nDISPLAY ', '\nSTOP RUN.', '\n', ' ', '']
        else:
            raise ValueError(f'Language {language} is not supported! Please choose from {list(Language)}')