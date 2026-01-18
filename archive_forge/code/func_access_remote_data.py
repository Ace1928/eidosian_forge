import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import requests
def access_remote_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': 'Unable to access data'}
    except requests.RequestException as e:
        return {'error': str(e)}