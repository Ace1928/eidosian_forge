import tkinter as tk
from tkinter import filedialog
import json


class UniversalGUIBuilder:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Universal GUI Builder")
        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.config = {"buttons": [], "labels": [], "entries": []}

        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="Save Configuration", command=self.save_configuration
        )
        self.file_menu.add_command(
            label="Load Configuration", command=self.load_configuration
        )
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.add_widgets_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.add_widgets_menu.add_command(
            label="Add Button", command=lambda: self.add_widget("button")
        )
        self.add_widgets_menu.add_command(
            label="Add Label", command=lambda: self.add_widget("label")
        )
        self.add_widgets_menu.add_command(
            label="Add Entry", command=lambda: self.add_widget("entry")
        )
        self.menu_bar.add_cascade(label="Add Widgets", menu=self.add_widgets_menu)

        self.preview_button = tk.Button(
            self.frame, text="Preview GUI", command=self.preview_gui
        )
        self.preview_button.pack(side=tk.BOTTOM, pady=10)

    def add_widget(self, widget_type: str):
        new_window = tk.Toplevel(self.master)
        new_window.title(f"Add {widget_type.capitalize()}")

        label = tk.Label(new_window, text=f"Enter {widget_type} properties:")
        label.pack(side=tk.TOP, pady=10)

        name_label = tk.Label(new_window, text="Name:")
        name_label.pack(side=tk.TOP, pady=5)

        name_entry = tk.Entry(new_window)
        name_entry.pack(side=tk.TOP, pady=5)

        if widget_type != "label":
            command_label = tk.Label(new_window, text="Command (function name):")
            command_label.pack(side=tk.TOP, pady=5)

            command_entry = tk.Entry(new_window)
            command_entry.pack(side=tk.TOP, pady=5)

        save_button = tk.Button(
            new_window,
            text="Save",
            command=lambda: self.save_widget(
                new_window,
                widget_type,
                name_entry.get(),
                command_entry.get() if widget_type != "label" else "",
            ),
        )
        save_button.pack(side=tk.BOTTOM, pady=10)

    def save_widget(
        self, window: tk.Toplevel, widget_type: str, name: str, command: str
    ):
        if widget_type == "button":
            self.config["buttons"].append({"name": name, "command": command})
        elif widget_type == "label":
            self.config["labels"].append({"name": name})
        elif widget_type == "entry":
            self.config["entries"].append({"name": name, "command": command})

        window.destroy()

    def preview_gui(self):
        preview_window = tk.Toplevel(self.master)
        preview_window.title("GUI Preview")

        for button in self.config["buttons"]:
            tk.Button(
                preview_window,
                text=button["name"],
                command=lambda: print(f"Executing {button['command']}"),
            ).pack(pady=5)

        for label in self.config["labels"]:
            tk.Label(preview_window, text=label["name"]).pack(pady=5)

        for entry in self.config["entries"]:
            tk.Entry(preview_window).pack(pady=5)

    def save_configuration(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            with open(file_path, "w") as file:
                json.dump(self.config, file, indent=4)
            print(f"Configuration saved to {file_path}")

    def load_configuration(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                self.config = json.load(file)
            print(f"Configuration loaded from {file_path}")


def main():
    root = tk.Tk()
    app = UniversalGUIBuilder(root)
    root.mainloop()


if __name__ == "__main__":
    main()
