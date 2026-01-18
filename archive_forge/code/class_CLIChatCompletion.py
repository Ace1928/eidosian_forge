from __future__ import annotations
import sys
from typing import TYPE_CHECKING, List, Optional, cast
from argparse import ArgumentParser
from typing_extensions import Literal, NamedTuple
from ..._utils import get_client
from ..._models import BaseModel
from ...._streaming import Stream
from ....types.chat import (
from ....types.chat.completion_create_params import (
class CLIChatCompletion:

    @staticmethod
    def create(args: CLIChatCompletionCreateArgs) -> None:
        params: CompletionCreateParams = {'model': args.model, 'messages': [{'role': cast(Literal['user'], message.role), 'content': message.content} for message in args.message], 'n': args.n, 'temperature': args.temperature, 'top_p': args.top_p, 'stop': args.stop, 'stream': False}
        if args.stream:
            params['stream'] = args.stream
        if args.max_tokens is not None:
            params['max_tokens'] = args.max_tokens
        if args.stream:
            return CLIChatCompletion._stream_create(cast(CompletionCreateParamsStreaming, params))
        return CLIChatCompletion._create(cast(CompletionCreateParamsNonStreaming, params))

    @staticmethod
    def _create(params: CompletionCreateParamsNonStreaming) -> None:
        completion = get_client().chat.completions.create(**params)
        should_print_header = len(completion.choices) > 1
        for choice in completion.choices:
            if should_print_header:
                sys.stdout.write('===== Chat Completion {} =====\n'.format(choice.index))
            content = choice.message.content if choice.message.content is not None else 'None'
            sys.stdout.write(content)
            if should_print_header or not content.endswith('\n'):
                sys.stdout.write('\n')
            sys.stdout.flush()

    @staticmethod
    def _stream_create(params: CompletionCreateParamsStreaming) -> None:
        stream = cast(Stream[ChatCompletionChunk], get_client().chat.completions.create(**params))
        for chunk in stream:
            should_print_header = len(chunk.choices) > 1
            for choice in chunk.choices:
                if should_print_header:
                    sys.stdout.write('===== Chat Completion {} =====\n'.format(choice.index))
                content = choice.delta.content or ''
                sys.stdout.write(content)
                if should_print_header:
                    sys.stdout.write('\n')
                sys.stdout.flush()
        sys.stdout.write('\n')