"""
completion.py

This module defines the API endpoints for generating text completions and embeddings using the RWKV model.
It includes endpoints for initiating chat completions, generating text completions, and creating embeddings.

Author: Your Name <your.email@example.com>
Created: YYYY-MM-DD
Last Updated: YYYY-MM-DD

Functions:
    eval_rwkv: Asynchronously evaluates the RWKV model for generating completions.
    chat_template_old: Generates a chat template for the old chat interface.
    chat_template: Generates a chat template for the current chat interface.
    chat_completions: Endpoint for generating chat completions.
    completions: Endpoint for generating text completions.
    embeddings: Endpoint for generating embeddings.

Classes:
    Role: Enum for user roles in chat.
    Message: Represents a message in the chat.
    ChatCompletionBody: Pydantic model for chat completion request body.
    CompletionBody: Pydantic model for completion request body.
    EmbeddingsBody: Pydantic model for embeddings request body.

Dependencies:
    asyncio, json, threading, typing, enum, fastapi, sse_starlette, pydantic, tiktoken, utils.rwkv, utils.log, global_var
"""

import asyncio
import json
from threading import Lock
from typing import List, Union
from enum import Enum
import base64

from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import tiktoken
from ..utils.rwkv import *
from ..utils.log import quick_log
import global_var

router = APIRouter()


class Role(Enum):
    """
    Enum class for defining roles within the chat.

    Attributes:
        User: Represents a user role in the chat.
        Assistant: Represents an assistant role in the chat.
        System: Represents a system role in the chat.
    """

    User = "user"
    Assistant = "assistant"
    System = "system"


class Message(BaseModel):
    """
    Represents a message in the chat.

    Attributes:
        role (Role): The role of the message sender (User, Assistant, System).
        content (str): The content of the message.
        raw (bool): Whether to treat content as raw text, described by a boolean flag.
    """

    role: Role
    content: str = Field(min_length=0)
    raw: bool = Field(False, description="Whether to treat content as raw text")


default_stop = [
    "\n\nUser",
    "\n\nQuestion",
    "\n\nQ",
    "\n\nHuman",
    "\n\nBob",
    "\n\nAssistant",
    "\n\nAnswer",
    "\n\nA",
    "\n\nBot",
    "\n\nAlice",
]


class ChatCompletionBody(ModelConfigBody):
    """
    Pydantic model for the request body of chat completions.

    Attributes:
        messages (Union[List[Message], None]): A list of Message objects or None.
        model (Union[str, None]): The model identifier, defaulting to "rwkv".
        stream (bool): Flag indicating whether to stream the response.
        stop (Union[str, List[str], None]): Stop sequences for the model.
        user_name (Union[str, None]): Internal user name.
        assistant_name (Union[str, None]): Internal assistant name.
        system_name (Union[str, None]): Internal system name.
        presystem (bool): Flag indicating whether to insert default system prompt at the beginning.
    """

    messages: Union[List[Message], None]
    model: Union[str, None] = "rwkv"
    stream: bool = False
    stop: Union[str, List[str], None] = default_stop
    user_name: Union[str, None] = Field(
        None, description="Internal user name", min_length=1
    )
    assistant_name: Union[str, None] = Field(
        None, description="Internal assistant name", min_length=1
    )
    system_name: Union[str, None] = Field(
        None, description="Internal system name", min_length=1
    )
    presystem: bool = Field(
        True, description="Whether to insert default system prompt at the beginning"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": Role.User.value, "content": "hello", "raw": False}
                ],
                "model": "rwkv",
                "stream": True,
                "stop": None,
                "user_name": "Lloyd",
                "assistant_name": "EVIE",
                "system_name": "Aurora",
                "presystem": True,
                "max_tokens": 8100,
                "temperature": 1.6,
                "top_p": 0.25,
                "presence_penalty": 0.6,
                "frequency_penalty": 1.4,
            }
        }
    }


class CompletionBody(ModelConfigBody):
    """
    Pydantic model for the request body of text completions.

    Attributes:
        prompt (Union[str, List[str], None]): The input prompt for the model.
        model (Union[str, None]): The model identifier, defaulting to "rwkv".
        stream (bool): Flag indicating whether to stream the response.
        stop (Union[str, List[str], None]): Stop sequences for the model.
    """

    prompt: Union[str, List[str], None]
    model: Union[str, None] = "rwkv"
    stream: bool = True
    stop: Union[str, List[str], None] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "The following is an epic science fiction masterpiece that is immortalized, "
                + "with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n",
                "model": "rwkv",
                "stream": True,
                "stop": None,
                "max_tokens": 2000,
                "temperature": 1.6,
                "top_p": 0.4,
                "presence_penalty": 0.6,
                "frequency_penalty": 1.4,
            }
        }
    }


completion_lock = Lock()

active_requests_count = 2  # Tracks the number of active requests


async def eval_rwkv(
    model: AbstractRWKV,
    request: Request,
    body: ModelConfigBody,
    prompt: str,
    stream: bool,
    stop: Union[str, List[str], None],
    chat_mode: bool,
):
    """
    Asynchronously evaluates the RWKV model to generate completions.

    This function handles the generation of text completions or chat completions based on the provided model configuration.
    It supports streaming responses for real-time updates.

    Args:
        model (AbstractRWKV): The RWKV model instance.
        request (Request): The current request context.
        body (ModelConfigBody): The configuration for the model.
        prompt (str): The input prompt for the model.
        stream (bool): Whether to stream the response.
        stop (Union[str, List[str], None]): The stop sequence(s) for the model.
        chat_mode (bool): Whether the completion is for chat.

    Yields:
        str or dict: The generated completion or a chunk of the generated completion if streaming.
    """
    global active_requests_count
    active_requests_count += 1
    quick_log(
        request, None, "Start Waiting. RequestsNum: " + str(active_requests_count)
    )
    while completion_lock.locked():
        if await request.is_disconnected():
            active_requests_count -= 1
            print(f"{request.client} Stop Waiting (Lock)")
            quick_log(
                request,
                None,
                "Stop Waiting (Lock). RequestsNum: " + str(active_requests_count),
            )
            return
        await asyncio.sleep(0.1)
    else:
        with completion_lock:
            if await request.is_disconnected():
                active_requests_count -= 1
                print(f"{request.client} Stop Waiting (Lock)")
                quick_log(
                    request,
                    None,
                    "Stop Waiting (Lock). RequestsNum: " + str(active_requests_count),
                )
                return
            if body is not None:
                set_rwkv_config(model, body)
            print(get_rwkv_config(model))

            response, prompt_tokens, completion_tokens = "", 0, 0
            for response, delta, prompt_tokens, completion_tokens in model.generate(
                prompt,
                stop=stop,
            ):
                if await request.is_disconnected():
                    break
                if stream:
                    yield json.dumps(
                        {
                            "object": (
                                "chat.completion.chunk"
                                if chat_mode
                                else "text_completion"
                            ),
                            # "response": response,
                            "model": model.name,
                            "choices": [
                                (
                                    {
                                        "delta": {"content": delta},
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                    if chat_mode
                                    else {
                                        "text": delta,
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                )
                            ],
                        }
                    )
            # torch_gc()
            active_requests_count -= 1
            if await request.is_disconnected():
                print(f"{request.client} Stop Waiting")
                quick_log(
                    request,
                    body,
                    response
                    + "\nStop Waiting. RequestsNum: "
                    + str(active_requests_count),
                )
                return
            quick_log(
                request,
                body,
                response + "\nFinished. RequestsNum: " + str(active_requests_count),
            )
            if stream:
                yield json.dumps(
                    {
                        "object": (
                            "chat.completion.chunk" if chat_mode else "text_completion"
                        ),
                        # "response": response,
                        "model": model.name,
                        "choices": [
                            (
                                {
                                    "delta": {},
                                    "index": 0,
                                    "finish_reason": "stop",
                                }
                                if chat_mode
                                else {
                                    "text": "",
                                    "index": 0,
                                    "finish_reason": "stop",
                                }
                            )
                        ],
                    }
                )
                yield "[DONE]"
            else:
                yield {
                    "object": "chat.completion" if chat_mode else "text_completion",
                    # "response": response,
                    "model": model.name,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "choices": [
                        (
                            {
                                "message": {
                                    "role": Role.Assistant.value,
                                    "content": response,
                                },
                                "index": 0,
                                "finish_reason": "stop",
                            }
                            if chat_mode
                            else {
                                "text": response,
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        )
                    ],
                }


def chat_template_old(
    model: TextRWKV, body: ChatCompletionBody, interface: str, user: str, bot: str
):
    """
    Generates a chat template for the old chat interface.

    This function constructs a chat template based on the old interface specifications, incorporating user and bot names.

    Args:
        model (TextRWKV): The RWKV model instance configured for text generation.
        body (ChatCompletionBody): The request body containing chat completion parameters.
        interface (str): The interface string used to differentiate user and bot messages.
        user (str): The name of the user in the chat.
        bot (str): The name of the bot in the chat.

    Returns:
        str: The generated chat template.
    """
    is_raven = model.rwkv_type == RWKVType.Raven

    completion_text: str = ""
    basic_system: Union[str, None] = None
    if body.presystem:
        if (
            body.messages
            and len(body.messages) > 0
            and body.messages[0].role == Role.System
        ):
            basic_system = body.messages[0].content

        if basic_system is None:
            completion_text = (
                f"""
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
{bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. \
{bot} likes to tell {user} a lot about herself and her opinions. \
{bot} usually gives {user} kind, helpful and informative advices.\n
"""
                if is_raven
                else (
                    f"{user}{interface} hi\n\n{bot}{interface} Hi. "
                    + "I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
                )
            )
        else:
            if not body.messages[0].raw:
                basic_system = (
                    basic_system.replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .replace("\n\n", "\n")
                    .replace("\n", " ")
                    .strip()
                )
            completion_text = (
                (
                    f"The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. "
                    if is_raven
                    else f"{user}{interface} hi\n\n{bot}{interface} Hi. "
                )
                + basic_system.replace("You are", f"{bot} is" if is_raven else "I am")
                .replace("you are", f"{bot} is" if is_raven else "I am")
                .replace("You're", f"{bot} is" if is_raven else "I'm")
                .replace("you're", f"{bot} is" if is_raven else "I'm")
                .replace("You", f"{bot}" if is_raven else "I")
                .replace("you", f"{bot}" if is_raven else "I")
                .replace("Your", f"{bot}'s" if is_raven else "My")
                .replace("your", f"{bot}'s" if is_raven else "my")
                .replace("你", f"{bot}" if is_raven else "我")
                + "\n\n"
            )

    for message in body.messages[(0 if basic_system is None else 1) :]:
        append_message: str = ""
        if message.role == Role.User:
            append_message = f"{user}{interface} " + message.content
        elif message.role == Role.Assistant:
            append_message = f"{bot}{interface} " + message.content
        elif message.role == Role.System:
            append_message = message.content
        if not message.raw:
            append_message = (
                append_message.replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\n\n", "\n")
                .strip()
            )
        completion_text += append_message + "\n\n"
    completion_text += f"{bot}{interface}"

    return completion_text


def chat_template(
    model: TextRWKV, body: ChatCompletionBody, interface: str, user: str, bot: str
):
    """
    Generates a chat template for the current chat interface.

    This function constructs a chat template based on the current interface specifications, incorporating user, bot, and system names.

    Args:
        model (TextRWKV): The RWKV model instance configured for text generation.
        body (ChatCompletionBody): The request body containing chat completion parameters.
        interface (str): The interface string used to differentiate user, bot, and system messages.
        user (str): The name of the user in the chat.
        bot (str): The name of the bot in the chat.

    Returns:
        str: The generated chat template.
    """
    completion_text: str = ""
    if body.presystem:
        completion_text = (
            f"{user}{interface} hi\n\n{bot}{interface} Hi. "
            + "I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n"
        )

    system = "System" if body.system_name is None else body.system_name
    for message in body.messages:
        append_message: str = ""
        if message.role == Role.User:
            append_message = f"{user}{interface} " + message.content
        elif message.role == Role.Assistant:
            append_message = f"{bot}{interface} " + message.content
        elif message.role == Role.System:
            append_message = f"{system}{interface} " + message.content
        completion_text += append_message + "\n\n"
    completion_text += f"{bot}{interface}"

    return completion_text


@router.post("/v1/chat/completions", tags=["Completions"])
@router.post("/chat/completions", tags=["Completions"])
async def chat_completions(body: ChatCompletionBody, request: Request):
    """
    Handles POST requests to generate chat completions.
    This endpoint supports both the new `/v1/chat/completions` and the legacy `/chat/completions` paths.

    Args:
        body (ChatCompletionBody): The request body containing the chat completion parameters.
        request (Request): The request object.

    Returns:
        EventSourceResponse or JSON: The generated chat completion or an error message.
    """
    model: TextRWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.messages is None or len(body.messages) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "messages not found")

    interface = model.interface
    user = model.user if body.user_name is None else body.user_name
    bot = model.bot if body.assistant_name is None else body.assistant_name

    if model.version < 5:
        completion_text = chat_template_old(model, body, interface, user, bot)
    else:
        completion_text = chat_template(model, body, interface, user, bot)

    user_code = model.pipeline.decode([model.pipeline.encode(user)[0]])
    bot_code = model.pipeline.decode([model.pipeline.encode(bot)[0]])
    if type(body.stop) == str:
        body.stop = [body.stop, f"\n\n{user_code}", f"\n\n{bot_code}"]
    elif type(body.stop) == list:
        body.stop.append(f"\n\n{user_code}")
        body.stop.append(f"\n\n{bot_code}")
    elif body.stop is None:
        body.stop = default_stop

    if body.stream:
        return EventSourceResponse(
            eval_rwkv(
                model, request, body, completion_text, body.stream, body.stop, True
            )
        )
    else:
        try:
            return await eval_rwkv(
                model, request, body, completion_text, body.stream, body.stop, True
            ).__anext__()
        except StopAsyncIteration:
            return None


@router.post("/v1/completions", tags=["Completions"])
@router.post("/completions", tags=["Completions"])
async def completions(body: CompletionBody, request: Request):
    """
    Handles POST requests to generate text completions.
    This endpoint supports both the new `/v1/completions` and the legacy `/completions` paths.

    Args:
        body (CompletionBody): The request body containing the prompt and model configuration.
        request (Request): The request object.

    Returns:
        EventSourceResponse or JSON: The generated completion or an error message.
    """
    model: AbstractRWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.prompt is None or body.prompt == "" or body.prompt == []:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "prompt not found")

    prompt = body.prompt if isinstance(body.prompt, str) else " ".join(body.prompt)

    if body.stream:
        return EventSourceResponse(
            eval_rwkv(model, request, body, prompt, body.stream, body.stop, False)
        )
    else:
        try:
            return await eval_rwkv(
                model, request, body, prompt, body.stream, body.stop, False
            ).__anext__()
        except StopAsyncIteration:
            return None


class EmbeddingsBody(BaseModel):
    """
    Pydantic model for the request body of embeddings.

    Attributes:
        input (Union[str, List[str], List[List[int]], None]): The input text or list of texts for which embeddings are to be generated.
        model (Union[str, None]): The model identifier, defaulting to "rwkv".
        encoding_format (str): The format of the encoding, can be None or "base64".
        fast_mode (bool): Flag indicating whether to use fast mode for generating embeddings.
    """

    input: Union[str, List[str], List[List[int]], None]
    model: Union[str, None] = "rwkv"
    encoding_format: str = None
    fast_mode: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "input": "a big apple",
                "model": "rwkv",
                "encoding_format": None,
                "fast_mode": False,
            }
        }
    }


def embedding_base64(embedding: List[float]) -> str:
    """
    Encodes a list of floating-point numbers representing an embedding into a base64 string.

    This function is used to convert embeddings into a base64 encoded string for efficient transmission over network protocols.

    Args:
        embedding (List[float]): The embedding to be encoded.

    Returns:
        str: The base64 encoded string of the embedding.
    """
    import numpy as np

    embedding_array = np.array(embedding, dtype=np.float32)
    return base64.b64encode(embedding_array.tobytes()).decode("utf-8")


@router.post("/v1/embeddings", tags=["Embeddings"])
@router.post("/embeddings", tags=["Embeddings"])
@router.post("/v1/engines/text-embedding-ada-002/embeddings", tags=["Embeddings"])
@router.post("/engines/text-embedding-ada-002/embeddings", tags=["Embeddings"])
async def embeddings(body: EmbeddingsBody, request: Request):
    """
    Handles POST requests to generate embeddings.
    This endpoint supports both the new `/v1/embeddings` and the legacy `/embeddings` paths, as well as specific engine paths.

    Args:
        body (EmbeddingsBody): The request body containing the input text(s) and model configuration.
        request (Request): The request object.

    Returns:
        JSON: The generated embeddings or an error message.
    """
    model: AbstractRWKV = global_var.get(global_var.Model)
    if model is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model not loaded")

    if body.input is None or body.input == "" or body.input == [] or body.input == [[]]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "input not found")

    global active_requests_count
    active_requests_count += 1
    quick_log(
        request, None, "Start Waiting. RequestsNum: " + str(active_requests_count)
    )
    while completion_lock.locked():
        if await request.is_disconnected():
            active_requests_count -= 1
            print(f"{request.client} Stop Waiting (Lock)")
            quick_log(
                request,
                None,
                "Stop Waiting (Lock). RequestsNum: " + str(active_requests_count),
            )
            return
        await asyncio.sleep(0.1)
    else:
        with completion_lock:
            if await request.is_disconnected():
                active_requests_count -= 1
                print(f"{request.client} Stop Waiting (Lock)")
                quick_log(
                    request,
                    None,
                    "Stop Waiting (Lock). RequestsNum: " + str(active_requests_count),
                )
                return

            base64_format = False
            if body.encoding_format == "base64":
                base64_format = True

            embeddings = []
            prompt_tokens = 0
            if type(body.input) == list:
                if type(body.input[0]) == list:
                    encoding = tiktoken.model.encoding_for_model(
                        "text-embedding-ada-002"
                    )
                    for i in range(len(body.input)):
                        if await request.is_disconnected():
                            break
                        input = encoding.decode(body.input[i])
                        embedding, token_len = model.get_embedding(
                            input, body.fast_mode
                        )
                        prompt_tokens += token_len
                        if base64_format:
                            embedding = embedding_base64(embedding)
                        embeddings.append(embedding)
                else:
                    for i in range(len(body.input)):
                        if await request.is_disconnected():
                            break
                        embedding, token_len = model.get_embedding(
                            body.input[i], body.fast_mode
                        )
                        prompt_tokens += token_len
                        if base64_format:
                            embedding = embedding_base64(embedding)
                        embeddings.append(embedding)
            else:
                embedding, prompt_tokens = model.get_embedding(
                    body.input, body.fast_mode
                )
                if base64_format:
                    embedding = embedding_base64(embedding)
                embeddings.append(embedding)

            active_requests_count -= 1
            if await request.is_disconnected():
                print(f"{request.client} Stop Waiting")
                quick_log(
                    request,
                    None,
                    "Stop Waiting. RequestsNum: " + str(active_requests_count),
                )
                return
            quick_log(
                request,
                None,
                "Finished. RequestsNum: " + str(active_requests_count),
            )

            ret_data = [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding,
                }
                for i, embedding in enumerate(embeddings)
            ]

            return {
                "object": "list",
                "data": ret_data,
                "model": model.name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens,
                },
            }
