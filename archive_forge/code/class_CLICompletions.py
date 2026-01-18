from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Optional, cast
from argparse import ArgumentParser
from functools import partial
from openai.types.completion import Completion
from .._utils import get_client
from ..._types import NOT_GIVEN, NotGivenOr
from ..._utils import is_given
from .._errors import CLIError
from .._models import BaseModel
from ..._streaming import Stream
class CLICompletions:

    @staticmethod
    def create(args: CLICompletionCreateArgs) -> None:
        if is_given(args.n) and args.n > 1 and args.stream:
            raise CLIError("Can't stream completions with n>1 with the current CLI")
        make_request = partial(get_client().completions.create, n=args.n, echo=args.echo, stop=args.stop, user=args.user, model=args.model, top_p=args.top_p, prompt=args.prompt, suffix=args.suffix, best_of=args.best_of, logprobs=args.logprobs, max_tokens=args.max_tokens, temperature=args.temperature, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)
        if args.stream:
            return CLICompletions._stream_create(cast(Stream[Completion], make_request(stream=True)))
        return CLICompletions._create(make_request())

    @staticmethod
    def _create(completion: Completion) -> None:
        should_print_header = len(completion.choices) > 1
        for choice in completion.choices:
            if should_print_header:
                sys.stdout.write('===== Completion {} =====\n'.format(choice.index))
            sys.stdout.write(choice.text)
            if should_print_header or not choice.text.endswith('\n'):
                sys.stdout.write('\n')
            sys.stdout.flush()

    @staticmethod
    def _stream_create(stream: Stream[Completion]) -> None:
        for completion in stream:
            should_print_header = len(completion.choices) > 1
            for choice in sorted(completion.choices, key=lambda c: c.index):
                if should_print_header:
                    sys.stdout.write('===== Chat Completion {} =====\n'.format(choice.index))
                sys.stdout.write(choice.text)
                if should_print_header:
                    sys.stdout.write('\n')
                sys.stdout.flush()
        sys.stdout.write('\n')