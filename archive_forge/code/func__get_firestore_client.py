from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _get_firestore_client() -> Client:
    try:
        import firebase_admin
        from firebase_admin import firestore
    except ImportError:
        raise ImportError('Could not import firebase-admin python package. Please install it with `pip install firebase-admin`.')
    try:
        firebase_admin.get_app()
    except ValueError as e:
        logger.debug('Initializing Firebase app: %s', e)
        firebase_admin.initialize_app()
    return firestore.client()