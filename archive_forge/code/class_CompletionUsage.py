from .._models import BaseModel
class CompletionUsage(BaseModel):
    completion_tokens: int
    'Number of tokens in the generated completion.'
    prompt_tokens: int
    'Number of tokens in the prompt.'
    total_tokens: int
    'Total number of tokens used in the request (prompt + completion).'