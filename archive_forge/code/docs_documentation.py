import json
from typing import Any, Dict, Optional
from fastapi.encoders import jsonable_encoder
from starlette.responses import HTMLResponse
from typing_extensions import Annotated, Doc  # type: ignore [attr-defined]

    Generate the HTML response with the OAuth2 redirection for Swagger UI.

    You normally don't need to use or change this.
    