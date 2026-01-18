import langchain
from langchain.llms import Replicate
from flask import Flask
from flask import request
import os
import requests
import json
class WhatsAppClient:
    API_URL = 'https://graph.facebook.com/v17.0/'
    WHATSAPP_API_TOKEN = '<Temporary access token from your WhatsApp API Setup>'
    WHATSAPP_CLOUD_NUMBER_ID = '<Phone number ID from your WhatsApp API Setup>'

    def __init__(self):
        self.headers = {'Authorization': f'Bearer {self.WHATSAPP_API_TOKEN}', 'Content-Type': 'application/json'}
        self.API_URL = self.API_URL + self.WHATSAPP_CLOUD_NUMBER_ID

    def send_text_message(self, message, phone_number):
        payload = {'messaging_product': 'whatsapp', 'to': phone_number, 'type': 'text', 'text': {'preview_url': False, 'body': message}}
        response = requests.post(f'{self.API_URL}/messages', json=payload, headers=self.headers)
        print(response.status_code)
        assert response.status_code == 200, 'Error sending message'
        return response.status_code