from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Callable
from . import rules_block
from .ruler import Ruler
from .rules_block.state_block import StateBlock
from .token import Token
from .utils import EnvType
class ParserBlock:
    """
    ParserBlock#ruler -> Ruler

    [[Ruler]] instance. Keep configuration of block rules.
    """

    def __init__(self) -> None:
        self.ruler = Ruler[RuleFuncBlockType]()
        for name, rule, alt in _rules:
            self.ruler.push(name, rule, {'alt': alt})

    def tokenize(self, state: StateBlock, startLine: int, endLine: int) -> None:
        """Generate tokens for input range."""
        rules = self.ruler.getRules('')
        line = startLine
        maxNesting = state.md.options.maxNesting
        hasEmptyLines = False
        while line < endLine:
            state.line = line = state.skipEmptyLines(line)
            if line >= endLine:
                break
            if state.sCount[line] < state.blkIndent:
                break
            if state.level >= maxNesting:
                state.line = endLine
                break
            for rule in rules:
                if rule(state, line, endLine, False):
                    break
            state.tight = not hasEmptyLines
            line = state.line
            if line - 1 < endLine and state.isEmpty(line - 1):
                hasEmptyLines = True
            if line < endLine and state.isEmpty(line):
                hasEmptyLines = True
                line += 1
                state.line = line

    def parse(self, src: str, md: MarkdownIt, env: EnvType, outTokens: list[Token]) -> list[Token] | None:
        """Process input string and push block tokens into `outTokens`."""
        if not src:
            return None
        state = StateBlock(src, md, env, outTokens)
        self.tokenize(state, state.line, state.lineMax)
        return state.tokens