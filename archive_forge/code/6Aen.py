import os
import subprocess
import pyudev
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def detect_usb_devices():
    """
    Detects connected USB devices using pyudev.

    Returns:
        list: A list of dictionaries containing model, manufacturer, and serial of each USB device.
    """
    try:
        context = pyudev.Context()
        devices = []
        for device in context.list_devices(subsystem="usb", DEVTYPE="usb_device"):
            if "ID_MODEL" in device:
                devices.append(
                    {
                        "model": device.get("ID_MODEL"),
                        "manufacturer": device.get("ID_VENDOR"),
                        "serial": device.get("ID_SERIAL_SHORT"),
                    }
                )
        return devices
    except Exception as e:
        logging.error(f"Error detecting USB devices: {e}")
        return []


def detect_adb_devices():
    """
    Detects connected ADB devices using adb command.

    Returns:
        list: A list of connected ADB device IDs.
    """
    try:
        result = subprocess.run(
            ["adb", "devices"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        devices = result.stdout.decode().strip().split("\n")[1:]
        connected_devices = [
            line.split("\t")[0] for line in devices if "device" in line
        ]
        return connected_devices
    except subprocess.CalledProcessError as e:
        logging.error(f"Error detecting ADB devices: {e.stderr.decode()}")
        return []


def get_device_info(device_id):
    """
    Retrieves device information using adb getprop command.

    Args:
        device_id (str): The ID of the ADB device.

    Returns:
        dict: A dictionary containing device properties.
    """
    try:
        device_info = {}
        result = subprocess.run(
            ["adb", "-s", device_id, "shell", "getprop"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        props = result.stdout.decode().strip().split("\n")
        for prop in props:
            key, value = prop.split(": ")
            device_info[key.strip()] = value.strip()
        return device_info
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting device info: {e.stderr.decode()}")
        return {}


def install_tools(device_info):
    """
    Installs necessary tools based on device information.

    Args:
        device_info (dict): A dictionary containing device properties.
    """
    try:
        if (
            "ro.product.manufacturer" in device_info
            and device_info["ro.product.manufacturer"] == "Samsung"
        ):
            subprocess.run(
                ["sudo", "apt", "install", "-y", "samsung-tools"], check=True
            )
        subprocess.run(["sudo", "apt", "install", "-y", "adb", "fastboot"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing tools: {e.stderr.decode()}")


def user_prompt(message):
    """
    Prompts the user with a yes/no question.

    Args:
        message (str): The message to display to the user.

    Returns:
        bool: True if the user responds with 'y', False otherwise.
    """
    return input(f"{message} (y/n): ").lower() == "y"


def setup_device(device_id, custom_rom_path=None):
    """
    Sets up the device by unlocking the bootloader, installing Magisk, and optionally installing a custom ROM.

    Args:
        device_id (str): The ID of the ADB device.
        custom_rom_path (str, optional): The path to the custom ROM file. Defaults to None.
    """
    try:
        info = get_device_info(device_id)
        install_tools(info)
        logging.info(f"Preparing device {device_id}...")
        if user_prompt("Would you like to unlock the bootloader?"):
            subprocess.run(["adb", "-s", device_id, "reboot", "bootloader"], check=True)
            subprocess.run(["fastboot", "-s", device_id, "oem", "unlock"], check=True)
        if user_prompt("Would you like to install Magisk?"):
            subprocess.run(
                ["adb", "-s", device_id, "install", "magisk.apk"], check=True
            )
        if custom_rom_path:
            logging.info(f"Installing custom ROM from {custom_rom_path}...")
            subprocess.run(
                ["adb", "-s", device_id, "sideload", custom_rom_path], check=True
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error setting up device: {e.stderr.decode()}")


class RootingTool(QtWidgets.QMainWindow):
    def __init__(self):
        super(RootingTool, self).__init__()
        uic.loadUi(
            "/home/lloyd/Downloads/PythonScripts/rootingtool/rooting_tool.ui", self
        )

        # Connect buttons to functions
        self.detectDevicesButton.clicked.connect(self.detect_devices)
        self.setupDeviceButton.clicked.connect(self.setup_device)
        self.selectRomButton.clicked.connect(self.select_rom)
        self.browseRomButton.clicked.connect(self.browse_rom)

        # Connect help menu action
        self.actionDocumentation.triggered.connect(self.show_documentation)

        # Initialize custom ROM path
        self.custom_rom_path = None

    def detect_devices(self):
        """
        Detects and lists all connected USB and ADB devices.
        """
        self.usbDevicesList.clear()
        usb_devices = detect_usb_devices()
        for device in usb_devices:
            self.usbDevicesList.addItem(
                f"Model: {device['model']}, Manufacturer: {device['manufacturer']}, Serial: {device['serial']}"
            )

        self.adbDevicesList.clear()
        adb_devices = detect_adb_devices()
        for device in adb_devices:
            self.adbDevicesList.addItem(device)

    def select_rom(self):
        """
        Opens a file dialog to select a custom ROM file for installation.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom ROM",
            "",
            "ZIP Files (*.zip);;All Files (*)",
            options=options,
        )
        if file_name:
            self.custom_rom_path = file_name
            self.romFileLabel.setText(f"Selected ROM: {os.path.basename(file_name)}")

    def browse_rom(self):
        """
        Opens a web browser to allow the user to browse for a custom ROM file.
        """
        self.browser_window.load(QUrl("https://www.xda-developers.com/"))
        self.browser_window.show()

    def setup_device(self):
        """
        Sets up the selected ADB device, optionally installing the selected custom ROM.
        """
        selected_device = self.adbDevicesList.currentItem().text()
        setup_device(selected_device, self.custom_rom_path)

    def show_documentation(self):
        """
        Displays the documentation for the Rooting Tool.
        """
        doc_message = (
            "Rooting Tool Documentation\n\n"
            "Detect Devices: Detects and lists all connected USB and ADB devices.\n"
            "Select Custom ROM: Opens a file dialog to select a custom ROM file for installation.\n"
            "Browse for Custom ROM: Opens a web browser to browse for a custom ROM file.\n"
            "Setup Device: Sets up the selected ADB device, optionally installing the selected custom ROM.\n"
            "\nSteps to use the tool:\n"
            "1. Click 'Detect Devices' to list connected devices.\n"
            "2. (Optional) Click 'Select Custom ROM' to choose a ROM file.\n"
            "3. (Optional) Click 'Browse for Custom ROM' to browse for a ROM file online.\n"
            "4. Select an ADB device from the list and click 'Setup Device' to begin the setup process.\n"
            "\nFor further assistance, refer to the official documentation or contact support."
        )
        QMessageBox.information(self, "Documentation", doc_message)


def main():
    """
    Main function to run the Rooting Tool application.
    """
    app = QtWidgets.QApplication(sys.argv)
    window = RootingTool()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
