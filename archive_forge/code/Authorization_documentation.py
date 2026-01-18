from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.AuthorizationApplication
import github.GithubObject
from github.GithubObject import Attribute, NotSet, Opt, _NotSetType

        :calls: `PATCH /authorizations/{id} <https://docs.github.com/en/developers/apps/authorizing-oauth-apps>`_
        :param scopes: list of string
        :param add_scopes: list of string
        :param remove_scopes: list of string
        :param note: string
        :param note_url: string
        :rtype: None
        