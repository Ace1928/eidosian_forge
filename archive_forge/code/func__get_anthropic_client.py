from typing import Any, List
def _get_anthropic_client() -> Any:
    try:
        import anthropic
    except ImportError:
        raise ImportError('Could not import anthropic python package. This is needed in order to accurately tokenize the text for anthropic models. Please install it with `pip install anthropic`.')
    return anthropic.Anthropic()