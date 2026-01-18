from collections import defaultdict
from logging import getLogger
from typing import Any, DefaultDict
from pip._vendor.resolvelib.reporters import BaseReporter
from .base import Candidate, Requirement
class PipDebuggingReporter(BaseReporter):
    """A reporter that does an info log for every event it sees."""

    def starting(self) -> None:
        logger.info('Reporter.starting()')

    def starting_round(self, index: int) -> None:
        logger.info('Reporter.starting_round(%r)', index)

    def ending_round(self, index: int, state: Any) -> None:
        logger.info('Reporter.ending_round(%r, state)', index)

    def ending(self, state: Any) -> None:
        logger.info('Reporter.ending(%r)', state)

    def adding_requirement(self, requirement: Requirement, parent: Candidate) -> None:
        logger.info('Reporter.adding_requirement(%r, %r)', requirement, parent)

    def rejecting_candidate(self, criterion: Any, candidate: Candidate) -> None:
        logger.info('Reporter.rejecting_candidate(%r, %r)', criterion, candidate)

    def pinning(self, candidate: Candidate) -> None:
        logger.info('Reporter.pinning(%r)', candidate)