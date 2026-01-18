import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
class SmsBackupReader(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('SMS Backup Reader')
        self.geometry('400x250')
        self.resizable(False, False)
        self.create_widgets()

    def create_widgets(self):
        frame_file = ttk.Frame(self)
        frame_file.pack(padx=10, pady=10, fill='x')
        ttk.Label(frame_file, text='Select sms.db:').pack(side='left')
        self.file_path_var = tk.StringVar()
        ttk.Entry(frame_file, textvariable=self.file_path_var, state='readonly').pack(side='left', expand=True, fill='x')
        ttk.Button(frame_file, text='Browse', command=self.open_file_dialog).pack(side='right')
        frame_contact = ttk.Frame(self)
        frame_contact.pack(padx=10, pady=10, fill='x')
        ttk.Label(frame_contact, text='Contact Number:').pack(side='left')
        self.contact_var = tk.StringVar()
        ttk.Entry(frame_contact, textvariable=self.contact_var).pack(side='right', expand=True, fill='x')
        frame_format = ttk.Frame(self)
        frame_format.pack(padx=10, pady=10, fill='x')
        ttk.Label(frame_format, text='Output Format:').pack(side='left')
        self.format_var = tk.StringVar(value='csv')
        ttk.Combobox(frame_format, textvariable=self.format_var, values=['csv', 'json', 'txt'], state='readonly').pack(side='right', expand=True, fill='x')
        ttk.Button(self, text='Extract Messages', command=self.extract_messages).pack(pady=10)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(title='Select the sms.db file', filetypes=[('Database files', '*.db')])
        self.file_path_var.set(file_path)

    def extract_messages(self):
        if not self.file_path_var.get():
            messagebox.showinfo('Error', 'Please select the sms.db file.')
            return
        if not self.contact_var.get():
            messagebox.showinfo('Error', 'Please enter the contact number.')
            return
        with connect_to_database(self.file_path_var.get()) as conn:
            extract_messages(conn, self.contact_var.get(), self.format_var.get())